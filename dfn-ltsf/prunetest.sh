#!/bin/bash
model_name=DLinear
seq_len=336
lr=0.01

# lr=0.001

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_$seq_len'_'96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --enc_in 8 \
#   --des 'Exp' \
#   --itr 1 --batch_size 8 --learning_rate $lr\
#   --apprx 1


python run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate $lr \
  --apprx 1

#   # --func_transfer 1 \
#   # --train_only True 



# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_$seq_len'_'96 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate $lr\
#   --apprx 1


# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_$seq_len'_'96 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.05\
#   --apprx 0


# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_$seq_len'_'96 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 8 --learning_rate 0.0001\
#   --dfn_exp 1\
#   --apprx 0
#   --apprx 1


  
# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_$seq_len'_'96 \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.001\
#   --apprx 1






# seq_len=104
# model_name=DLinear

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path national_illness.csv \
#   --model_id national_illness_$seq_len'_'24 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 18 \
#   --pred_len 24 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.01 \
#   --apprx 1


python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.05 



python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16 \
  --apprx 1