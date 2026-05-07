#!/bin/bash
#SBATCH -J vevo_vq4096
#SBATCH --time=48:00:00
#SBATCH -o logs/vq4096_%j.out
#SBATCH -e logs/vq4096_%j.err
#SBATCH -p amd_h100_2
#SBATCH --gres=gpu:1                  # GPU 1개 사용
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

source /scratch/$USER/miniconda3/bin/activate vevo
export HF_HOME=/scratch/$USER/.cache/huggingface

echo "=== Vevo Content-Style Tokenizer (Vocab 4096) Training Start ==="
sh egs/codec/vevo/fvq4096.sh
echo "=== Training Finished ==="