
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

for seed in {1..50}
do
    # echo "Starting iteration with seed $seed at $(date)"
    # seq_len=336

    # python run_longExp.py \
    # --is_training 1 \
    # --root_path ./dataset/ \
    # --data_path electricity.csv \
    # --model_id expSeed_Electricity_$seq_len'_'96 \
    # --model $model_name \
    # --data custom \
    # --features M \
    # --seq_len $seq_len \
    # --pred_len 96 \
    # --enc_in 321 \
    # --des 'Exp' \
    # --itr 1 --batch_size 16  --learning_rate $lr \
    # --apprx 1 \
    # --apprx_target $apprx_tgt \
    # --seed $seed \
    # --soft_flag 0 \
    # --n_func $n_func \
    # --beta_alter 0 \
    # --alpha_mult 1 \
    # --func_transfer 1 \
    # --alter_all 0 \
    # --apprx_epochs 100
 echo "Starting iteration with seed $seed at $(date)"
    seq_len=336

    python run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path exchange_rate.csv \
    --model_id expSeed_Exchange_$seq_len'_'96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len 96 \
    --enc_in 321 \
    --des 'Exp' \
    --itr 1 --batch_size 8  --learning_rate 0.001 \
    --apprx 1 \
    --apprx_target $apprx_tgt \
    --seed $seed \
    --soft_flag 0 \
    --n_func $n_func \
    --beta_alter 0 \
    --alpha_mult 1 \
    --func_transfer 1 \
    --alter_all 0 \
    --apprx_epochs 150
done