#!/bin/bash
#SBATCH --job-name=g2p_es_v4
#SBATCH --output=logs/es_v4_%j.out
#SBATCH --error=logs/es_v4_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

mkdir -p logs
source ~/.bashrc
conda activate g2p
cd $SLURM_SUBMIT_DIR

# Spanish is highly regular so less dropout and augmentation needed
python src/run.py \
    --lang es --strategy dedup \
    --output_dir runs/es_dedup_v4 \
    --d_model 384 --num_heads 6 --num_enc_layers 6 --num_dec_layers 6 --d_ff 1024 \
    --dropout 0.1 --layer_drop 0.05 \
    --batch_size 128 --grad_accumulation 2 \
    --num_epochs 120 --peak_lr 3e-4 --warmup_epochs 5 \
    --patience 15 --dropout_anneal_epoch 20 --dropout_target 0.02 \
    --augment --augment_prob 0.08 --use_cache --beam_size 4 \
    --save_top_k 3 --num_workers 8
