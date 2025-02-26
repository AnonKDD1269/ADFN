#!/bin/bash

type="mnist"
epoch=10
lr=0.1
fig_name="refrun_mnist"
batch_size=128

for seed in 1 2 3 ; do
    python train_cls_v2.py --mode ref --epoch $epoch --type $type --lr $lr --layer 3 --batch_size $batch_size --fig_name $fig_name --seed $seed
done

