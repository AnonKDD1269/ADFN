#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
model_name=DLinear
lr=0.01
apprx=1
seed=2021
apprx_tgt=seasonal



# national_illness
seq_len=104
lr=0.01
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
  --learning_rate 0.01 \
  --apprx $apprx \
  --apprx_target $apprx_tgt \
  --seed $seed

# #   # Traffic
# seq_len=336
# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path traffic.csv \
#   --model_id traffic_$seq_len'_'96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --enc_in 862 \
#   --des 'Exp' \
#   --itr 1 --batch_size 16 --learning_rate 0.05 \
#   --apprx $apprx \
#   --seed $seed

# # # weather
# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path weather.csv \
#   --model_id weather_$seq_len'_'96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --enc_in 21 \
#   --des 'Exp' \
#   --itr 1 --batch_size 16 \
#   --apprx $apprx