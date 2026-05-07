#!/bin/bash
#SBATCH -J vevo_infer_all        # 작업명 지정 (짧은 옵션 -J 사용)
#SBATCH -p amd_a100nv_8          # 파티션 지정 (짧은 옵션 -p 사용)
#SBATCH --nodes=1                # 작업 수행 노드 수
#SBATCH --ntasks-per-node=1      # 노드당 프로세스 수 (파이썬 실행 1개)
#SBATCH --cpus-per-task=8        # 프로세스당 할당될 CPU 코어 수
#SBATCH --gres=gpu:1             # [매우 중요!!] GPU 1장 할당 (찾으신 목록엔 없지만 필수입니다)
#SBATCH --time=02:00:00          # 최대 작업 수행 시간 (2시간)
#SBATCH -o logs/infer_%j.out     # 출력 로그 (-o)
#SBATCH -e logs/infer_%j.err     # 에러 로그 (-e)
#SBATCH --comment=pytorch        # KISTI 통계용 애플리케이션명 (필수 요구 사항인 경우 대비)

source /scratch/$USER/miniconda3/bin/activate vevo
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/Amphion
mkdir -p logs

echo "=== 1. Vevo-Timbre ==="
python -m models.vc.vevo.infer_vevotimbre

echo "=== 2. Vevo-Style ==="
python -m models.vc.vevo.infer_vevostyle

echo "=== 3. Vevo-Voice ==="
python -m models.vc.vevo.infer_vevovoice

echo "=== 4. Vevo-TTS ==="
python -m models.vc.vevo.infer_vevotts

echo "=== All Inferences Completed ==="