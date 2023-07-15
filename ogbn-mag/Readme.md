# HGAMLP on Ogbn-mag

## Training without extra embeddings

python main.py --stages 400 400 400 500  --num-hops 2 --label-feats --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --n-layers-3 3 --act leaky_relu --bns --label-bns --lr 0.002 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --amp --gpu 0 --seeds 0

For the first time this command is executed, the dataset ogbn-mag will be automatically downloaded in the folder `../data/`.

The codes generate random initialized embeddings for node types author/topic/institution.

## Training with extra embeddings from ComplEx

**1.Generate extra embeddings from ComplEx**

Please make sure that the ogbn-mag dataset has been downloaded in the folder `../data/`.

Then under the folder `../data/complex_nars`, run

```setup
python convert_to_triplets.py --dataset mag
bash train_graph_emb.sh mag
```

Check the running log to find where the generated ComplEx features are saved. For example, if the save folder is `ckpts/ComplEx_mag_0`, run

```setup
python split_node_emb.py --dataset mag --emb-file ckpts/ComplEx_mag_0/mag_ComplEx_entity.npy
```

**2.Training our HGAMLP model**

python main.py --stages 400 400 400 500  --num-hops 2 --label-feats --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --n-layers-3 3 --extra-embedding ComplexE --act leaky_relu --bns --label-bns --lr 0.002 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --amp --gpu 0 --seeds 0
