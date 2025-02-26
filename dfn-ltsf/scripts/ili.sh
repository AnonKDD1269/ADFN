#!/bin/bash
# # Create directories if they do not exist
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=104
model_name=DLinear
apprx_tgt=all
n_func=1
seed=2021

# for seed in {1..100}
# do

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id national_illness_${seq_len}_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 24 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --apprx 1 \
  --apprx_target $apprx_tgt \
  --seed $seed \
  --soft_flag 0 \
  --n_func $n_func \
  --beta_alter 0 \
  --alpha_mult 1 \
  --func_transfer 1 \
  --alter_all 1 \
  --apprx_epochs 100
  # --div_check 1


python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id national_illness_${seq_len}_36 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 36 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.01 \
  --apprx 1 \
  --apprx_target $apprx_tgt \
  --seed $seed \
  --soft_flag 0 \
  --n_func $n_func \
  --beta_alter 0 \
  --alpha_mult 1 \
  --func_transfer 1 \
  --alter_all 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id national_illness_${seq_len}_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 48 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.01 \
  --apprx 1 \
  --apprx_target $apprx_tgt \
  --seed $seed\
  --apprx 1 \
  --apprx_target $apprx_tgt \
  --seed $seed \
  --soft_flag 0 \
  --n_func $n_func \
  --beta_alter 0 \
  --alpha_mult 1 \
  --func_transfer 1 \
  --alter_all 1
  
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id national_illness_${seq_len}_60 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 60 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.01 \
  --apprx 1 \
  --apprx_target $apprx_tgt \
  --seed $seed \
  --soft_flag 0 \
  --n_func $n_func \
  --beta_alter 0 \
  --alpha_mult 1 \
  --func_transfer 1 \
  --alter_all 1
# # done