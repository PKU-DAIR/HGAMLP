# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
####此版本是加入embedding对齐之后###
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch_sparse import SparseTensor


class Transformer(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.query.weight, gain=gain)
        xavier_uniform_(self.key.weight, gain=gain)
        xavier_uniform_(self.value.weight, gain=gain)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]
        return o.permute(0,2,1,3).reshape((B, M, C)) + x
    
class Conv1d1x1(nn.Module):
    def __init__(self, cin, cout, groups, bias=True, cformat='channel-first'):
        super(Conv1d1x1, self).__init__()
        self.cin = cin
        self.cout = cout
        self.groups = groups
        self.cformat = cformat
        if not bias:
            self.bias = None
        if self.groups == 1: # different keypoints share same kernel
            self.W = nn.Parameter(torch.randn(self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(1, self.cout))
        else:
            self.W = nn.Parameter(torch.randn(self.groups, self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.groups, self.cout))

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.groups == 1:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,mn->bnc', x, self.W) + self.bias.T
            else:
                assert False
        else:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
            else:
                assert False

class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))          
        
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


class GlobalMetaAggregator(nn.Module):
    def __init__(self, feats_keys, extra_feat_keys, label_feat_keys, data_size, data_size_extra, in_feats, r_len, tgt_type, input_dropout, dropout, hidden, out_feats, n_layers_2, att_drop, enhance):
        super(GlobalMetaAggregator, self).__init__()
        self.data_size = data_size
        self.in_feats = in_feats
        self.r_len = r_len
        self.tgt_type = tgt_type
        self.feat_keys = sorted(feats_keys)
        self.extra_feat_keys = sorted(extra_feat_keys)
        self.label_feat_keys = sorted(label_feat_keys)
        num_channels = len(self.feat_keys) + len(self.extra_feat_keys)+len(self.label_feat_keys)
        self.input_drop = nn.Dropout(input_dropout)
        self.att_dropout = nn.Dropout(att_drop)
        self.prelu = nn.PReLU()
        self.enhance = enhance

        #################### for global vector###############
        self.embed_weight_r = nn.ParameterList()    ###for meta fusion
        for _ in range(r_len):
            weight = nn.Parameter(torch.Tensor(hidden, 1))
            gain = nn.init.calculate_gain("sigmoid")
            nn.init.xavier_uniform_(weight, gain=gain)
            self.embed_weight_r.append(weight)

        self.embed_weight_r_extra = nn.ParameterList()  ###for extra_meta fusion
        for _ in range(len(data_size_extra.keys())):
            weight = nn.Parameter(torch.Tensor(in_feats, 1))
            gain = nn.init.calculate_gain("sigmoid")
            nn.init.xavier_uniform_(weight, gain=gain)
            self.embed_weight_r_extra.append(weight)

        self.w = nn.Parameter(torch.Tensor(hidden, 1))   #####for r_ensemble
        gain = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(self.w, gain=gain)
        #################### for global vector###############

        ############for feature projection###################
        self.r_embedding = nn.ParameterList()
        for i in range(r_len):
            self.embedding = nn.ParameterDict({})
            for k, v in data_size.items():
                self.embedding[str(k)] = nn.Parameter(
                    torch.Tensor(v, in_feats).uniform_(-0.5, 0.5))
            self.r_embedding.append(self.embedding)
        ############for feature projection###################
        self.r_extra_embedding = nn.ParameterList()
        for i in range(r_len):
            self.extra_embedding = nn.ParameterDict({})
            for k, v in data_size_extra.items():
                self.extra_embedding[str(k)] = nn.Parameter(
                    torch.Tensor(v, in_feats).uniform_(-0.5, 0.5))
            self.r_extra_embedding.append(self.extra_embedding)
        
        self.r_label_embedding = nn.ParameterList()
        for i in range(r_len):
            self.label_embedding = nn.ParameterDict({})
            for k in self.label_feat_keys:
                self.label_embedding[k] = nn.Parameter(
                    torch.Tensor(out_feats, in_feats).uniform_(-0.5, 0.5))
            self.r_label_embedding.append(self.label_embedding)
        
        self.out_project = FeedForwardNet(
                hidden, hidden, out_feats, n_layers_2, dropout
            )

        self.layers = nn.Sequential(
            Conv1d1x1(in_feats, hidden, num_channels, bias=True, cformat='channel-first'),  ###KD2
            nn.LayerNorm([num_channels, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            Conv1d1x1(hidden, hidden, num_channels, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_channels, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        for layer in self.layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()
        
        if self.enhance:
            def add_nonlinear_layers(hidden, dropout, bns=False):
                if bns:
                    return [
                        nn.BatchNorm1d(hidden),
                        nn.PReLU(),
                        nn.Dropout(dropout)
                    ]
                else:
                    return [
                        nn.PReLU(),
                        nn.Dropout(dropout)
                    ]
            lr_output_layers = [
                [nn.Linear(hidden, hidden, bias=not False)] + add_nonlinear_layers(hidden, dropout, False)
                for _ in range(n_layers_2-1)]
            self.lr_output = nn.Sequential(*(
                [ele for li in lr_output_layers for ele in li] + [
                nn.Linear(hidden, out_feats, bias=False),
                nn.BatchNorm1d(out_feats)]))
            
            for layer in self.lr_output:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, raw_feats_ensemble, extra_features_buffer, r_label_dict):
        feats_ensemble = []
        for r, embedding, r_extra_features, extra_embedding, r_lable_features, labels_embeding in zip(raw_feats_ensemble, self.r_embedding, extra_features_buffer, self.r_extra_embedding, r_label_dict, self.r_label_embedding):
            if isinstance(r[self.tgt_type], SparseTensor):
                mapped_feats = {k: self.input_drop(x @ embedding[k[-1]]) for k, x in r.items()}
            elif isinstance(r[self.tgt_type], torch.Tensor):
                mapped_feats = {k: self.input_drop(x @ embedding[k]) for k, x in r.items()}

            mapped_feats_extra = {k: self.input_drop(x @ extra_embedding[k]) for k, x in r_extra_features.items()}
            mapped_label_feats = {k: self.input_drop(x @ labels_embeding[k]) for k, x in r_lable_features.items()}

            extra_feats_list = []
            for _, feats in mapped_feats_extra.items():
                extra_feats_list.append(feats)
            extra_feats = []

            #########enhance the extra features from de-redundancy scheme#######
            for feats, embed_weight in zip(extra_feats_list, self.embed_weight_r_extra):
                global_vector = torch.softmax(torch.sigmoid(torch.matmul(feats, embed_weight)).squeeze(2), dim=-1)
                output_r = 0
                for i in range(feats.shape[1]):
                    output_r = output_r + feats[:,i,:].mul(self.att_dropout(global_vector[:, i].unsqueeze(1)))
                extra_feats.append(output_r)
            #########enhance the extra features from de-redundancy scheme#######

            ###Se
            features = [mapped_feats[k] for k in self.feat_keys] + extra_feats + [mapped_label_feats[k] for k in self.label_feat_keys]
            features = torch.stack(features, dim=1)
            features = self.layers(features)            
            ###
            feats_ensemble.append(features)

        new_feats_r = []
        for feats, embed_weight in zip(feats_ensemble, self.embed_weight_r):
            new_feats = []
            global_vector = torch.softmax(torch.sigmoid(torch.matmul(feats, embed_weight)).squeeze(2), dim=-1)
            output_r = 0
            for i in range(feats.shape[1]):
                output_r = output_r + feats[:,i,:].mul(self.att_dropout(global_vector[:, i].unsqueeze(1)))
            new_feats_r.append(output_r)
        

        h = torch.stack(new_feats_r, dim=1)
        global_vector = torch.softmax(torch.sigmoid(torch.matmul(h, self.w)).squeeze(2), dim=-1)
        output_r_final = 0
        for i, hidden in enumerate(new_feats_r):
            output_r_final = output_r_final + hidden.mul(self.att_dropout(global_vector[:, i].unsqueeze(1)))

        if not self.enhance:
            output = self.out_project(output_r_final)
        else:
            output = self.lr_output(output_r_final)


        return output