#!/bin/bash
#SBATCH --job-name=qwen_rl
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4          # 2 DataLoader workers x 2 for headroom
#SBATCH --mem=32G                  # model + optimizer states + data
#SBATCH --time=1-00:00:00          # normal partition max; switch to gpu-long for more
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL       # remove if you don't want emails
#SBATCH --mail-user=fhongyi@YOUR_DOMAIN  # replace or remove
 
# ── Environment ───────────────────────────────────────────────────────────────
mkdir -p logs
 

# Useful diagnostics in the log
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU(s):    $CUDA_VISIBLE_DEVICES"
nvidia-smi
 
# ── Cache dirs (avoid filling home quota) ─────────────────────────────────────
export HF_HOME=~/cs4248_project/.hf_cache
export TRANSFORMERS_CACHE=~/cs4248_project/.hf_cache
export PYTHONUNBUFFERED=1

 
# ── Run ───────────────────────────────────────────────────────────────────────
uv run python rl.py
