export CUDA_VISIBLE_DEVICES=0

rates=(0.1)
seeds=(2)

model_name=Transformer
##############################################
for rate in "${rates[@]}"
do
  for seed in "${seeds[@]}"
  do
    python -u myrun.py \
      --task_name imputation \
      --is_training 1 \
      --root_path ./dataset/HD/ \
      --data_path HD.csv \
      --model_id HD_norev \
      --model $model_name \
      --data MyData \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 5 \
      --dec_in 5 \
      --c_out 5 \
      --batch_size 16 \
      --d_model 16 \
      --d_ff 32 \
      --des 'Exp' \
      --itr 1 \
      --top_k 3 \
      --learning_rate 0.001 \
      --mask_rate $rate \
      --seed $seed
  done
done

model_name=myTransformer
##############################################
for rate in "${rates[@]}"
do
  for seed in "${seeds[@]}"
  do
    python -u myrun.py \
      --task_name imputation \
      --is_training 1 \
      --root_path ./dataset/HD/ \
      --data_path HD.csv \
      --model_id HD_norev \
      --model $model_name \
      --data MyData \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 5 \
      --dec_in 5 \
      --c_out 5 \
      --batch_size 16 \
      --d_model 16 \
      --d_ff 32 \
      --des 'Exp' \
      --itr 1 \
      --top_k 3 \
      --learning_rate 0.001 \
      --mask_rate $rate \
      --seed $seed
  done
done