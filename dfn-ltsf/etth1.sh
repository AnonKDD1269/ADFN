
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
n_func=1
apprx_tgt=seasonal
alter_all=0

#naive
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len_again'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate $lr\
  --apprx 0 \
  --apprx_target all \
  --seed $seed \
  --soft_flag 0 \
  --n_func $n_func \
  --beta_alter 0 \
  --alpha_mult 1 \
  --func_transfer 1 \
  --alter_all 0
#naive

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len_again'_'192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate $lr \
  --apprx $apprx \
  --apprx_target $apprx_tgt \
  --seed $seed \
  --soft_flag 0 \
  --n_func $n_func \
  --beta_alter 0 \
  --alpha_mult 1 \
  --func_transfer 1 \
  --alter_all $alter_all \
  --apprx_epochs 100


python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len_again_again'_'192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate $lr \
  --apprx $apprx \
  --apprx_target trend \
  --seed $seed \
  --soft_flag 0 \
  --n_func $n_func \
  --beta_alter 0 \
  --alpha_mult 1 \
  --func_transfer 1 \
  --alter_all $alter_all \
  --apprx_epochs 100


python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len_again_again'_'192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate $lr \
  --apprx $apprx \
  --apprx_target all \
  --seed $seed \
  --soft_flag 0 \
  --n_func $n_func \
  --beta_alter 0 \
  --alpha_mult 1 \
  --func_transfer 1 \
  --alter_all 1 \
  --apprx_epochs 100

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_$seq_len_again_again'_'336 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 336 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate $lr \
#   --apprx 1 \
#   --apprx_target $apprx_tgt \
#   --seed $seed \
#   --soft_flag 0 \
#   --n_func $n_func \
#   --beta_alter 0 \
#   --alpha_mult 1 \
#   --func_transfer 1 \
#   --alter_all 1

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_$seq_len_again_again'_'720 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 720 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate $lr \
#   --apprx 1 \
#   --apprx_target $apprx_tgt \
#   --seed $seed \
#   --soft_flag 0 \
#   --n_func $n_func \
#   --beta_alter 0 \
#   --alpha_mult 1 \
#   --func_transfer 1 \
#   --alter_all 1