

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear
lr=0.01
apprx_tgt=seasonal
seed=2021
n_func=2

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
  --apprx 1 \
  --apprx_target $apprx_tgt \
  --seed $seed \
  --soft_flag 0 \
  --n_func $n_func \
  --beta_alter 0 \
  --alpha_mult 1 \
  --func_transfer 0 \
  --div_check 0 \
  --apprx_epochs 10
