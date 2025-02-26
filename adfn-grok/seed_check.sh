#!/bin/bash


# for seed in {1..50}
# do
#     python3 grok_apprx.py --DATA_SEED $seed --func_transfer 1 --from_pretrained 0 --apprx_epochs 10000
# done

python3 grok_apprx.py --DATA_SEED 598 --func_transfer 1 --from_pretrained 0 --apprx_epochs 25000
