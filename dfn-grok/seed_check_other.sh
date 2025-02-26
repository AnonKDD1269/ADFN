#!/bin/bash


for seed in {51..100}
do
    python3 grok_apprx.py --DATA_SEED $seed --func_transfer 1 --from_pretrained 1
done