#!/bin/bash
#SBATCH --job-name=sarcasm_inference
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fhongyi@YOUR_DOMAIN   # replace or remove

# ── Environment ───────────────────────────────────────────────────────────────
mkdir -p logs

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU(s):    $CUDA_VISIBLE_DEVICES"
nvidia-smi

# ── Cache dirs (avoid filling home quota) ─────────────────────────────────────
export HF_HOME=~/cs4248_project/.hf_cache
export TRANSFORMERS_CACHE=~/cs4248_project/.hf_cache
export PYTHONUNBUFFERED=1

# ── Run ───────────────────────────────────────────────────────────────────────
# All arguments passed to sbatch after '--' are forwarded to the script.
# Example overrides:
#   sbatch run_dataset_inference.sh -- --model_path ./my_checkpoint --batch_size 32
#   sbatch run_dataset_inference.sh -- --resume
uv run python dataset_inference.py \
    --base_model_path ./qwen_sarcasm_best \
    --adapter_path    ./qwen_grpo_rl-ed/checkpoint-7500 \
    --dataset_path    ./Sarcasm_Headlines_Dataset.json \
    --output_path     ./sarcastic_generated.jsonl \
    --batch_size      16 \
    --max_new_tokens  64 \
    --temperature     0 \
    --log_dir         logs \
    "$@"

