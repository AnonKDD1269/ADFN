
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi


# seq_len=192
model_name=DLinear
seq_len=336
lr=0.001
n_func=1
apprx_tgt=seasonal
alter_all=0
apprx=0
seed=2021

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
#   --itr 1 --batch_size 8 --learning_rate $lr \
#   --apprx $apprx \
#   --apprx_target $apprx_tgt \
#   --seed $seed \
#   --soft_flag 0 \
#   --n_func $n_func \
#   --beta_alter 0 \
#   --alpha_mult 1 \
#   --func_transfer 1 \
#   --alter_all 1

apprx_tgt=seasonal
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate $lr \
  --apprx $apprx \
  --apprx_target $apprx_tgt \
  --seed $seed \
  --soft_flag 0 \
  --n_func $n_func \
  --beta_alter 0 \
  --alpha_mult 1 \
  --func_transfer 1 \
  --alter_all 0 \
  --apprx_epochs 100

apprx=1

apprx_tgt=seasonal
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate $lr \
  --apprx $apprx \
  --apprx_target $apprx_tgt \
  --seed $seed \
  --soft_flag 0 \
  --n_func $n_func \
  --beta_alter 0 \
  --alpha_mult 1 \
  --func_transfer 1 \
  --alter_all 0 \
  --apprx_epochs 100


apprx_tgt=trend
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate $lr \
  --apprx $apprx \
  --apprx_target $apprx_tgt \
  --seed $seed \
  --soft_flag 0 \
  --n_func $n_func \
  --beta_alter 0 \
  --alpha_mult 1 \
  --func_transfer 1 \
  --alter_all 0 \
  --apprx_epochs 100



apprx_tgt=all
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate $lr \
  --apprx $apprx \
  --apprx_target $apprx_tgt \
  --seed $seed \
  --soft_flag 0 \
  --n_func $n_func \
  --beta_alter 0 \
  --alpha_mult 1 \
  --func_transfer 1 \
  --alter_all 1 \
  --apprx_epochs 150


# apprx_tgt=trend
# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_$seq_len'_'336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 336 \
#   --enc_in 8 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate $lr \
#   --apprx $apprx \
#   --apprx_target $apprx_tgt \
#   --seed $seed \
#   --soft_flag 0 \
#   --n_func $n_func \
#   --beta_alter 0 \
#   --alpha_mult 1 \
#   --func_transfer 1 \
#   --alter_all $alter_all

# apprx_tgt=all
# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_$seq_len'_'720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 720 \
#   --enc_in 8 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate $lr \
#   --apprx $apprx \
#   --apprx_target $apprx_tgt \
#   --seed $seed \
#   --soft_flag 0 \
#   --n_func $n_func \
#   --beta_alter 0 \
#   --alpha_mult 1 \
#   --func_transfer 1 \
#   --alter_all $alter_all