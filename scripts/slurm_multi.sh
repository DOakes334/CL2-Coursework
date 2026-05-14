#!/bin/bash
#SBATCH --job-name=g2p_multi_v4
#SBATCH --output=logs/multi_v4_%j.out
#SBATCH --error=logs/multi_v4_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

mkdir -p logs
source ~/.bashrc
conda activate g2p
cd $SLURM_SUBMIT_DIR

# Multilingual English + Spanish with language tags
python run.py \
    --lang en es --strategy dedup --multilingual \
    --output_dir runs/multi_v4 \
    --d_model 384 --num_heads 6 --num_enc_layers 6 --num_dec_layers 6 --d_ff 1024 \
    --dropout 0.15 --layer_drop 0.1 \
    --batch_size 128 --grad_accumulation 2 \
    --num_epochs 150 --peak_lr 3e-4 --warmup_epochs 8 \
    --patience 15 --dropout_anneal_epoch 25 --dropout_target 0.05 \
    --augment --augment_prob 0.15 --use_cache --beam_size 4 \
    --save_top_k 3 --num_workers 8
