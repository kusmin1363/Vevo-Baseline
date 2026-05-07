#!/bin/bash
#SBATCH -J vevo_vq32
#SBATCH --time=48:00:00               # 학습은 오래 걸리니 넉넉하게 48시간
#SBATCH -o logs/vq32_%j.out
#SBATCH -e logs/vq32_%j.err
#SBATCH -p amd_h100_2
#SBATCH --gres=gpu:1                # GPU 4개 사용 (필요시 8개로 수정)
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16            # 데이터 로딩 일꾼 넉넉하게 배정

source /scratch/$USER/miniconda3/bin/activate vevo
export HF_HOME=/scratch/$USER/.cache/huggingface

echo "=== Vevo Content Tokenizer (Vocab 32) Training Start ==="
sh egs/codec/vevo/fvq32.sh
echo "=== Training Finished ==="