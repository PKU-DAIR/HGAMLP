# HGAMLP on the four medium-scale datasets

## Training

To reproduce the results of HGAMLP on four medium-scale datasets, please run following commands.

For **DBLP**:

```bash
python3 train_hgb.py  --dataset DBLP  --num-hops 4  --num-hidden 64  --lr 0.001 --dropout 0.5 --eval-every 1   --num-epochs 200 --input-dropout 0.1 --seed 1 --gpu 0 --threshold 2 --label-feats --num-label-hops 2 --r 0.0 0.2
```

For **ACM**:
```bash
python3 train_hgb.py  --dataset ACM  --num-hops 5 --num-hidden 64  --lr 0.001 --dropout 0.5 --eval-every 1   --num-epochs 200 --seed 1 --gpu 0 --threshold 1 --patience 10
```

For **IMDB**:

```bash
python train_hgb.py  --dataset IMDB  --num-hops 5 --num-hidden 512  --lr 0.001 --dropout 0.5 --eval-every 1   --num-epochs 200 --gpu 0 --threshold 5 --att-drop 0.5 --r 0.0 --ff-layer-2 3 --seed 11 --enhance --label-feats --num-label-hops 2
```

For **Freebase**:

```bash
python train_hgb.py  --dataset Freebase --num-hops 2  --num-hidden 512  --lr 0.001 --dropout 0.5 --eval-every 1  --batch-size 5000 --eval-batch-size 5000 --num-epochs 200  --threshold 1 --label-feats --num-label-hops 3 --att-drop 0.5
```