#!/bin/bash

type="mnist"
epoch=10
lr=0.1
fig_name="consistency"
batch_size=128

for seed in {1..100}; do
    python train_cls_v2.py --epoch $epoch --type $type --lr $lr --layer 3 --batch_size $batch_size --fig_name $fig_name --seed $seed
done

