#!/bin/bash

#SBATCH -J onenoise_emb_penalty1_tilefix
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem=256g
#SBATCH -t 06-00:00:00
#SBATCH -p 3090-gcondo --gres=gpu:8
#SBATCH --export=CXX=g++
#SBATCH -o /users/amaesumi/logs/onenoise_emb_penalty1_tilefix.out
#SBATCH -e /users/amaesumi/logs/onenoise_emb_penalty1_tilefix.err

cd /users/amaesumi/OneNoise_dev

module load anaconda/2023.09-0-7nso27y
source /gpfs/runtime/opt/anaconda/2023.03-1/etc/profile.d/conda.sh
conda activate noise

accelerate launch --config_file accelerate_8gpu.yml train.py \
    --exp_name emb_pen1_tilefix \
    --out_dir /users/amaesumi/data/amaesumi/noise_exp/ \
    --data_dir /users/amaesumi/data/prj_noise_share/noise_separate.hdf5 \
    --image_size 256 \
    --num_workers 16 \
    --batch_size 8 \
    --save_every 10000 \
    --sample_every 3000 \
    --sample_timesteps 50 \
    --grad_accum 2 \
    --spade True \
    --attention True \
    --model_config tiny \
    --pos_enc 0 \
    --optim adamw \
    --objective pred_noise \
    --cutmix 1 \
    --rebuttal True \
    --loss_fn l2 \
    --num_samples 16 \
    --emb_penalty 0.02
