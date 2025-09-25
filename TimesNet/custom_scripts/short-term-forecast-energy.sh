# train window 20 testsize 0.5
python3 -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id energy_rollin20_testsize0.5 \
  --model TimesNet \
  --data custom \
  --root_path ~/Dropbox/elvas-thesis/timesNet/data/ \
  --data_path timesnet_energy_var6_window20_testsize0.5_train.csv \
  --features M \
  --target ENEL_logret \
  --seq_len 48 --label_len 0 --pred_len 20 \
  --enc_in 6 --dec_in 6 --c_out 6 \
  --freq b \
  --d_model 64 --d_ff 128 --e_layers 1 --n_heads 2 \
  --top_k 3 --num_kernels 3 --dropout 0.1 \
  --batch_size 8 --train_epochs 10 --patience 3 \
  --learning_rate 5e-4 \
  --num_workers 0 --use_gpu True --gpu 0
  
  
# train window 20 testsize 0.2
python3 -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id energy_rollin20_testsize0.2 \
  --model TimesNet \
  --data custom \
  --root_path ~/Dropbox/elvas-thesis/timesNet/data/ \
  --data_path timesnet_energy_var6_window20_testsize0.2_train.csv \
  --features M \
  --target ENEL_logret \
  --seq_len 48 --label_len 0 --pred_len 20 \
  --enc_in 6 --dec_in 6 --c_out 6 \
  --freq b \
  --d_model 64 --d_ff 128 --e_layers 1 --n_heads 2 \
  --top_k 3 --num_kernels 3 --dropout 0.1 \
  --batch_size 8 --train_epochs 10 --patience 3 \
  --learning_rate 5e-4 \
  --num_workers 0 --use_gpu True --gpu 0
  
  
# train window 1 testsize 0.2
python3 -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id energy_nextday_testsize0.2 \
  --model TimesNet \
  --data custom \
  --root_path ~/Dropbox/elvas-thesis/timesNet/data/ \
  --data_path timesnet_energy_var6_window1_testsize0.2_train.csv \
  --features M \
  --target ENEL_logret \
  --seq_len 48 --label_len 0 --pred_len 1 \
  --enc_in 6 --dec_in 6 --c_out 6 \
  --freq b \
  --d_model 64 --d_ff 128 --e_layers 1 --n_heads 2 \
  --top_k 3 --num_kernels 3 --dropout 0.1 \
  --batch_size 8 --train_epochs 10 --patience 3 \
  --learning_rate 5e-4 \
  --num_workers 0 --use_gpu True --gpu 0
  



# all the 20d testsize 0.5 settings excluding various exogs. 

python3 -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id energy_nointerest \
  --model TimesNet \
  --data custom \
  --root_path ~/Dropbox/elvas-thesis/timesNet/data/excluding-exogs \
  --data_path timesnet_energy_var6_window20_testsize0.5_nointerest_train.csv \
  --features M \
  --target ENEL_logret \
  --seq_len 48 --label_len 0 --pred_len 20 \
  --enc_in 5 --dec_in 5 --c_out 5 \
  --freq b \
  --d_model 64 --d_ff 128 --e_layers 1 --n_heads 2 \
  --top_k 3 --num_kernels 3 --dropout 0.1 \
  --batch_size 8 --train_epochs 10 --patience 3 \
  --learning_rate 5e-4 \
  --num_workers 0 --use_gpu True --gpu 0


