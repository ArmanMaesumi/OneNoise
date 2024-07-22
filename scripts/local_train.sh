cd ..

python train.py \
    --data_dir ./data \
    --lr 8e-5 \
    --batch_size 8 \
    --grad_accum 2 \
    --sample_every 5000 \
    --sample_timesteps 50 \
    --model_config tiny \
    --emb_penalty 0.02 \
    --num_workers 16