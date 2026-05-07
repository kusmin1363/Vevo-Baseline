#!/bin/bash
#SBATCH -J cleanup
#SBATCH --time=00:02:00
#SBATCH -o logs/cleanup_%j.out
#SBATCH -e logs/cleanup_%j.err
#SBATCH -p amd_h100_2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=gpu59
#SBATCH --cpus-per-task=1
#SBATCH --comment=pytorch
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16          


echo "=== Cleanup on $(hostname) at $(date) ==="

echo ""
echo "[Before] My processes on this node:"
ps -u $USER -o pid,etime,cmd --no-headers | grep -E "python|sh egs" | grep -v grep

echo ""
echo "Killing all my python processes..."
pkill -u $USER python
sleep 2

# 안 죽으면 강제로
echo "Force killing remaining processes..."
pkill -9 -u $USER python
sleep 1

echo ""
echo "[After] My processes on this node:"
ps -u $USER -o pid,etime,cmd --no-headers | grep -E "python|sh egs" | grep -v grep

echo ""
echo "=== Cleanup done at $(date) ==="