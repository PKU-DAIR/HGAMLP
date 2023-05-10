import argparse
import time
from utils_hgb import check_acc
import numpy as np
import torch
import torch.nn as nn
import logging
import uuid
from model_hgb import *
import sys
from data_hgb import *
from utils_hgb import *
import random
import torch.nn.functional as F
import datetime

def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.gpu < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}"
    print("seed", args.seed)
    # Load dataset
    g, adjs, features_list_dict, labels, num_classes, dl, train_nid, val_nid, test_nid \
        = load_dataset(args)
    g = None

    # =======
    # neighbor aggregation
    # =======
    if args.dataset == 'DBLP':
        tgt_type = 'A'
    elif args.dataset == 'ACM':
        tgt_type = 'P'
    elif args.dataset == 'IMDB':
        tgt_type = 'M'
    elif args.dataset == 'Freebase':
        tgt_type = '0'
    else:
        assert 0
    max_length = args.num_hops + 1
    #r_list = [0.]
    r_list = args.r
    print(r_list)
    feats_r_ensemble = []
    extra_feats_r_ensemble = []
    label_feats_r_ensemble = []
    features_list_dict_cp = features_list_dict
    with torch.no_grad():
        for r in r_list:
            #############normalization#########################
            for k in adjs.keys():
                adjs[k].storage._value = None
            for k in adjs.keys():
                row_sum = adjs[k].sum(dim=1)
                col_sum = adjs[k[1]+k[0]].sum(dim=1)
                norm_left = row_sum.pow(r-1).flatten()
                norm_left[torch.isinf(norm_left)] = 0.
                norm_left = norm_left.unsqueeze(1)
                norm_right = col_sum.pow(-r).flatten()
                norm_right[torch.isinf(norm_right)] = 0.
                d_mat_norm_right = SparseTensor(row=torch.arange(norm_right.shape[0]), col=torch.arange(norm_right.shape[0]), value = norm_right, sparse_sizes=(norm_right.shape[0], norm_right.shape[0]))
                adjs[k].storage._value = (adjs[k].mul(norm_left)).to(device).matmul(d_mat_norm_right.to(device)).to('cpu').storage._value
            if args.dataset != "Freebase":
                prop_device = 'cuda:{}'.format(args.gpu)
                threshold_metalen = args.threshold
                features_list_dict, extra_features_buffer = pre_feature_prop(adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, threshold_metalen, prop_device, args.enhance, prop_feats=True, echo=True)                    
                feats = {}
                feats_extra = {}
                keys = list(features_list_dict.keys())
                print(f'For tgt {tgt_type}, feature keys {keys}')
                keys_extra = list(extra_features_buffer.keys())
                print(f'For tgt {tgt_type}, extra feature keys {keys_extra}')
                for k in keys:
                    feats[k] = features_list_dict.pop(k)
                for k in keys_extra:
                    feats_extra[k] = extra_features_buffer.pop(k)
                data_size = {k: v.size(-1) for k, v in feats.items()}
                data_size_extra = {k: v.size(-1) for k, v in feats_extra.items()}

            elif args.dataset == "Freebase":
                prop_device = 'cuda:{}'.format(args.gpu)
                threshold_metalen = args.threshold
                
                features_list_dict, extra_features_buffer = pre_feature_prop_freebase(adjs, threshold_metalen, tgt_type, args.num_hops, max_length, prop_device, args.enhance, prop_feats=True, echo=True)
                feats = {}
                feats_extra = {}
                keys = list(features_list_dict.keys())
                print(f'For tgt {tgt_type}, feature keys {keys}')
                keys_extra = list(extra_features_buffer.keys())
                for k in keys:
                    feats[k] = features_list_dict.pop(k)
                for k in keys_extra:
                    feats_extra[k] = extra_features_buffer.pop(k)
                feats['0'] = SparseTensor.eye(dl.nodes['count'][0])
                print(f'For tgt {tgt_type}, extra feature keys {keys_extra}')
                data_size = dict(dl.nodes['count'])
                data_size_extra = {k: v.size(-1) for k, v in feats_extra.items()}

            feats_r_ensemble.append(feats)
            extra_feats_r_ensemble.append(feats_extra)

            # =======
            # labels propagate alongside the metapath
            # 
            # =======
            num_nodes = dl.nodes['count'][0]
            label_feats = {}
            if args.label_feats:
                if args.dataset != 'IMDB':
                    label_onehot = torch.zeros((num_nodes, num_classes))
                    label_onehot[train_nid] = F.one_hot(labels[train_nid], num_classes).float()
                else:
                    label_onehot = torch.zeros((num_nodes, num_classes))

                max_length_label = args.num_label_hops + 1

                print(f'Current label-prop num hops = {args.num_label_hops}')
                # compute k-hop feature
                prop_tic = datetime.datetime.now()
                if args.dataset == 'Freebase' and args.num_label_hops <= args.num_hops:
                    meta_adjs = pre_label_feats(
                                adjs, tgt_type, args.num_label_hops, max_length_label, prop_feats=False, echo=True, prop_device=prop_device)
                    meta_adjs = {k: v for k, v in meta_adjs.items() if k[-1] == '0' and len(k) < max_length_label}
                else:
                    if args.dataset == 'Freebase':
                        meta_adjs = pre_label_feats(
                                adjs, tgt_type, args.num_label_hops, max_length_label, prop_feats=False, echo=True, prop_device=prop_device)
                    else:
                        meta_adjs = pre_label_feats(
                            adjs, tgt_type, args.num_label_hops, max_length_label, prop_feats=False, echo=True, prop_device=prop_device)

                if args.dataset == 'Freebase':
                    left_keys = ['00', '000', '0000', '0010', '0030', '0040', '0050', '0060', '0070']
                    remove_keys = list(set(list(meta_adjs.keys())) - set(left_keys))
                    for k in remove_keys:
                        meta_adjs.pop(k)

                    label_onehot_g = label_onehot.to(prop_device)
                    for k, v in meta_adjs.items():
                        if args.dataset != 'Freebase':
                            label_feats[k] = remove_diag(v) @ label_onehot
                        else:
                            label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to('cpu')

                    del label_onehot_g
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    for k, v in meta_adjs.items():
                        if args.dataset != 'Freebase':
                            label_feats[k] = remove_diag(v) @ label_onehot
                        else:
                            label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to('cpu')

                    gc.collect()

                    if args.dataset == 'IMDB':
                        condition = lambda ra,rb,rc,k: True
                        check_acc(label_feats, condition, labels, train_nid, val_nid, test_nid, show_test=False, loss_type='bce')
                    else:
                        condition = lambda ra,rb,rc,k: True
                        check_acc(label_feats, condition, labels, train_nid, val_nid, test_nid, show_test=True)
                print('Involved label keys', label_feats.keys())

                prop_toc = datetime.datetime.now()
                print(f'Time used for label prop {prop_toc - prop_tic}')

            label_feats_r_ensemble.append(label_feats)

    if args.dataset == 'IMDB':
        labels = labels.float().to(device)
    labels = labels.to(device)
    
    # Set up logging
    logging.basicConfig(format='[%(levelname)s] %(message)s',
                        level=logging.INFO)

    r_len = len(r_list)

    import os
    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)
    checkpt_file = checkpt_folder + uuid.uuid4().hex
    print('checkpt_file', checkpt_file)

    # Create model
    in_feats = 512
    model = GlobalMetaAggregator(feats.keys(), feats_extra.keys(), label_feats.keys(), data_size, data_size_extra, in_feats,
                                  r_len, tgt_type, args.input_dropout, args.dropout, args.num_hidden, 
                                  num_classes, args.ff_layer_2, args.att_drop, args.enhance)

    logging.info("# Params: {}".format(get_n_params(model)))
    model.to(device)
    print(model)
    if len(labels.shape) == 1:
        loss_fcn = nn.CrossEntropyLoss()
    else:
        loss_fcn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    # Start training

    best_epoch = 0
    best_val_loss = 1000000
    best_test_loss = 0
    best_val_micro = 0
    best_test_micro = 0
    best_val_macro = 0
    best_test_macro = 0
    time_sum = 0
    for epoch in range(1, args.num_epochs + 1):
            start = time.time()
            train(model, feats_r_ensemble, extra_feats_r_ensemble, label_feats_r_ensemble, labels, train_nid, loss_fcn, optimizer, args.batch_size, args.dataset)
            end = time.time()
            with torch.no_grad():
                train_micro, val_micro, test_micro, train_macro, val_macro, test_macro, loss_train, loss_val, loss_test = test(
                    model, feats_r_ensemble, extra_feats_r_ensemble, label_feats_r_ensemble, labels, train_nid, val_nid, test_nid, loss_fcn, args.eval_batch_size, args.dataset)  
            #end = time.time()
            if epoch>1:
                time_sum = time_sum + (end - start)
            if epoch % 1 == 0:
                log = "Epoch {}, Times(s): {:.4f}".format(epoch, end - start)
                log += ", mac,mic: Tra({:.4f} {:.4f}), Val({:.4f} {:.4f}), Tes({:.4f} {:.4f}) Val_loss({:.4f})".format(train_macro, train_micro, val_macro, val_micro, test_macro, test_micro, loss_val)
                logging.info(log)
            #if (args.dataset != 'Freebase' and args.dataset != 'IMDB' and loss_val <= best_val_loss) or (args.dataset == 'Freebase' and sum([val_micro, val_macro]) >= sum([best_val_micro, best_val_macro])) or (args.dataset == 'IMDB' and sum([val_micro, val_macro]) >= sum([best_val_micro, best_val_macro])): #:
            if (args.dataset != 'IMDB' and loss_val <= best_val_loss) or (args.dataset == 'IMDB' and sum([val_micro, val_macro]) >= sum([best_val_micro, best_val_macro])):
                best_epoch = epoch
                best_val_loss = loss_val
                best_test_loss = loss_test
                best_val_micro = val_micro
                best_val_macro = val_macro
                best_test_micro = test_micro
                best_test_macro = test_macro
                torch.save(model.state_dict(), f'{checkpt_file}.pkl')
            if args.dataset == 'ACM' or args.dataset == 'DBLP':
                if epoch - best_epoch > args.patience:
                    break
            elif args.dataset == 'IMDB':
                if epoch - best_epoch > args.patience: break
            elif args.dataset == 'Freebase':
                if epoch - best_epoch > args.patience: break
    print(f'Best Epoch {best_epoch} at {checkpt_file.split("/")[-1]}\n\t with val loss {best_val_loss:.4f} and test loss {best_test_loss:.4f}')
    logging.info("macro: Best Val {:.4f}, Best Test {:.4f}".format(best_val_macro, best_test_macro))
    logging.info("micro: Best Val {:.4f}, Best Test {:.4f}".format(best_val_micro, best_test_micro))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HGAMLP")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-hidden", type=int, default=512)
    parser.add_argument("--num-hops", type=int, default=2,
                        help="number of hops")
    parser.add_argument("--label-feats", action='store_true', default=False,
                        help="whether to use the label propagated features")
    parser.add_argument("--num-label-hops", type=int, default=2,
                        help="number of hops for propagation of raw features")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dataset", type=str, default="DBLP")
    parser.add_argument("--root", type=str, default='../data/')
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50000)   ##50000
    parser.add_argument("--eval-batch-size", type=int, default=25000,  ##250000
                        help="evaluation batch size, -1 for full batch")
    parser.add_argument("--ff-layer-2", type=int, default=2,
                        help="number of feed-forward layers")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu-preprocess", action="store_true",
                        help="Preprocess on CPU")
    parser.add_argument("--in-feats", type=int, default=512)
    parser.add_argument('--patience', type=int, default=20, help='Patience.')
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to add residual branch the raw input features")
    parser.add_argument("--input-dropout", type=float, default=0.1,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.5,
                        help="att dropout of attention scores")
    parser.add_argument("--enhance", action='store_true', default=False)
    parser.add_argument("--ACM-keep-F", type=bool, default=False)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--r", nargs='+', type=float, default=[0.0],
                        help="the seed used in the training")
    args = parser.parse_args()

    print(args)
    main(args)