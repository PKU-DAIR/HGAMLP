# HGAMLP: A Scalable Training Framework for Heterogeneous Graph Neural Networks

## Requirements

#### 1. Neural network libraries for GNNs

* [pytorch](https://pytorch.org/get-started/locally/)
* [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
* [dgl](https://www.dgl.ai/pages/start.html)

## Data preparation

For experiments in Motivation section and on four medium-scale datasets, please download datasets `DBLP.zip`, `ACM.zip`, `IMDB.zip`, `Freebase.zip` from [the source of HGB benchmark](https://cloud.tsinghua.edu.cn/d/a2728e52cd4943efa389/), and extract content from these compresesed files under the folder `'./data/'`.

For experiments on the large dataset ogbn-mag, the dataset will be automatically downloaded from OGB challenge.
