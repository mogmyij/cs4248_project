"""
dataset_inference.py

Batch inference script: reads non-sarcastic headlines from
Sarcasm_Headlines_Dataset.json, generates sarcastic rewrites using
the GRPO-trained model, and writes output to a JSONL file whose
format matches deepseek_sarcastic.jsonl:
  {"input": "<original headline>", "output": "<sarcastic rewrite>"}

Designed to run under SLURM via run_dataset_inference.sh.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    log_path = Path(log_dir) / f"inference_{job_id}.log"

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_path}")
    return logger


# ---------------------------------------------------------------------------
# Environment diagnostics
# ---------------------------------------------------------------------------

def log_environment(logger: logging.Logger) -> None:
    logger.info("=" * 60)
    logger.info("ENVIRONMENT DIAGNOSTICS")
    logger.info("=" * 60)
    logger.info(f"SLURM_JOB_ID      : {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    logger.info(f"SLURMD_NODENAME   : {os.environ.get('SLURMD_NODENAME', 'N/A')}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")
    logger.info(f"Python            : {sys.version}")
    logger.info(f"PyTorch           : {torch.__version__}")
    try:
        import transformers
        logger.info(f"Transformers      : {transformers.__version__}")
    except ImportError:
        pass

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"GPU {i}: {props.name}  "
                f"{props.total_memory / 1024**3:.1f} GB"
            )
        # nvidia-smi summary
        ret = os.system("nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader 2>/dev/null")
        if ret != 0:
            logger.warning("nvidia-smi not available")
    else:
        logger.warning("No CUDA devices detected — running on CPU (will be very slow)")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_non_sarcastic(dataset_path: str, logger: logging.Logger) -> list[str]:
    logger.info(f"Loading dataset from: {dataset_path}")
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    total = len(data)
    non_sarcastic = [item["headline"] for item in data if item["is_sarcastic"] == 0]
    logger.info(
        f"Dataset: {total} total entries, "
        f"{len(non_sarcastic)} non-sarcastic selected"
    )
    return non_sarcastic


def count_existing_lines(path: str) -> int:
    """Count completed lines in an existing JSONL output file."""
    p = Path(path)
    if not p.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------------------
# Prompt builder (matches rl.py exactly)
# ---------------------------------------------------------------------------

def build_prompt(headline: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a headline writer for a satirical news publication like the Onion. "
        "Transform the given genuine news headline into a sarcastic version, styled like the Onion.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Rewrite this headline: {headline}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(base_model_path: str, adapter_path: str, logger: logging.Logger):
    # Tokenizer lives in the base SFT model folder
    logger.info(f"Loading tokenizer from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # Left-padding is required for batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    logger.info(f"Loading base model from: {base_model_path}  (dtype={dtype})")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    logger.info(f"Base model loaded in {time.time() - t0:.1f}s")

    logger.info(f"Applying LoRA adapter from: {adapter_path}")
    t1 = time.time()
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    logger.info(f"Adapter applied in {time.time() - t1:.1f}s")

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Total parameters: {num_params:.0f}M  (total load: {time.time() - t0:.1f}s)")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory after model load — allocated: {alloc:.2f} GB, reserved: {reserved:.2f} GB")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_batch(
    model,
    tokenizer,
    headlines: list[str],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    prompts = [build_prompt(h) for h in headlines]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    results = []
    for seq in outputs:
        decoded = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        # Take only the first line, strip whitespace (matches rl.py cleanup)
        first_line = decoded.split("\n")[0].strip()
        results.append(first_line)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch sarcastic headline inference")
    parser.add_argument("--base_model_path", default="./qwen_sarcasm_best",
                        help="Path to the SFT base model (default: ./qwen_sarcasm_best)")
    parser.add_argument("--adapter_path", default="./qwen_grpo_finetuned_final",
                        help="Path to the GRPO LoRA adapter (default: ./qwen_grpo_finetuned_final)")
    parser.add_argument("--dataset_path", default="./Sarcasm_Headlines_Dataset.json",
                        help="Path to the input dataset JSON")
    parser.add_argument("--output_path", default="./sarcastic_generated.jsonl",
                        help="Path for the output JSONL file")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generation (default: 16)")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Max new tokens to generate per headline (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature; 0 = greedy (default: 0.7)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing output file, skipping already-generated lines")
    parser.add_argument("--log_dir", default="logs",
                        help="Directory for log files (default: logs/)")
    args = parser.parse_args()

    logger = setup_logging(args.log_dir)
    log_environment(logger)

    logger.info("=" * 60)
    logger.info("INFERENCE CONFIGURATION")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k:<22} = {v}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    headlines = load_non_sarcastic(args.dataset_path, logger)

    skip = 0
    if args.resume:
        skip = count_existing_lines(args.output_path)
        if skip > 0:
            logger.info(f"Resume mode: skipping {skip} already-generated headlines")
            headlines = headlines[skip:]
        else:
            logger.info("Resume mode: no existing output found, starting from scratch")

    total = len(headlines)
    if total == 0:
        logger.info("Nothing to generate. Exiting.")
        return

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(args.base_model_path, args.adapter_path, logger)

    # ------------------------------------------------------------------
    # Estimate throughput
    # ------------------------------------------------------------------
    logger.info(f"Starting generation: {total} headlines, batch_size={args.batch_size}")
    logger.info(
        f"Estimated batches: {(total + args.batch_size - 1) // args.batch_size}"
    )

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------
    t_start = time.time()
    generated_count = 0
    errors = 0

    output_file = open(args.output_path, "a", encoding="utf-8")
    try:
        pbar = tqdm(
            range(0, total, args.batch_size),
            desc="Generating",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_start in pbar:
            batch_headlines = headlines[batch_start: batch_start + args.batch_size]

            try:
                batch_outputs = generate_batch(
                    model, tokenizer, batch_headlines,
                    args.max_new_tokens, args.temperature,
                )
            except Exception as exc:
                logger.error(
                    f"Batch starting at index {skip + batch_start} failed: {exc}"
                )
                # Write empty outputs so resume offsets stay consistent
                batch_outputs = [""] * len(batch_headlines)
                errors += len(batch_headlines)

            for headline, output in zip(batch_headlines, batch_outputs):
                record = json.dumps({"input": headline, "output": output}, ensure_ascii=False)
                output_file.write(record + "\n")

            output_file.flush()
            generated_count += len(batch_headlines)

            # Periodic progress log every 500 headlines
            if generated_count % 500 < args.batch_size:
                elapsed = time.time() - t_start
                rate = generated_count / elapsed if elapsed > 0 else 0
                eta_sec = (total - generated_count) / rate if rate > 0 else float("inf")
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec)) if eta_sec != float("inf") else "unknown"

                mem_str = ""
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    mem_str = f"  GPU mem: {alloc:.2f} GB"

                logger.info(
                    f"Progress: {generated_count + skip}/{total + skip} headlines  "
                    f"({rate:.1f} headlines/s  ETA: {eta_str}){mem_str}"
                )

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received — flushing output and exiting gracefully")
    finally:
        output_file.flush()
        output_file.close()

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    output_size = Path(args.output_path).stat().st_size / 1024**2 if Path(args.output_path).exists() else 0

    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Headlines generated : {generated_count}")
    logger.info(f"Errors              : {errors}")
    logger.info(f"Time elapsed        : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    logger.info(
        f"Throughput          : {generated_count / elapsed:.1f} headlines/s"
        if elapsed > 0 else "Throughput: N/A"
    )
    logger.info(f"Output file         : {args.output_path}  ({output_size:.2f} MB)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

