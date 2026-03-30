"""
Qwen2.5 SFT — Sarcastic Headline Generator
==========================================
Converts neutral news headlines into sarcastic/satirical versions
using supervised fine-tuning on a decoder-only causal LM.
"""

import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-1.5B"
OUTPUT_DIR     = "./qwen_sarcasm_"
DATA_PATH      = "./passed_pairs.jsonl"

MAX_INPUT_LEN  = 128    # prompt tokens
MAX_TARGET_LEN = 64     # response tokens

BATCH_SIZE     = 8
GRAD_ACCUM     = 4
EPOCHS         = 3
LR             = 2e-5
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.06
FP16           = False  # bfloat16 does not need a scaler; keep False

SYSTEM_PROMPT = (
    "You are a headline writer for a satirical news publication in the style of The Onion. "
    "Transform the given genuine news headline into a sarcastic, satirical version."
)

import multiprocessing
multiprocessing.set_start_method("fork")

# ── Dataset ───────────────────────────────────────────────────────────────────
class SarcasmDataset(Dataset):
    """
    Each sample is a full ChatML-formatted sequence:
        <|im_start|>system ...
        <|im_start|>user   Rewrite this headline: {input}
        <|im_start|>assistant {output}<|im_end|>

    Labels mirror input_ids but with prompt tokens masked to -100,
    so the loss is computed only on the assistant response tokens.
    """

    def __init__(self, dataset, tokenizer):
        self.data      = dataset.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data.iloc[idx]

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Rewrite this headline: {pair['input']}"},
            {"role": "assistant", "content": pair["output"]},
        ]

        # Tokenize the prompt portion alone to find the boundary
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1],               # system + user only
            add_generation_prompt=True,  # appends <|im_start|>assistant\n
            tokenize=False,
        )
        full_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )

        prompt_len = len(
            self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].squeeze()
        )

        full_enc = self.tokenizer(
            full_text,
            max_length=MAX_INPUT_LEN + MAX_TARGET_LEN,
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = full_enc["input_ids"].squeeze()
        attention_mask = full_enc["attention_mask"].squeeze()

        # Mask prompt tokens from loss; only the assistant reply is supervised
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── Collator ──────────────────────────────────────────────────────────────────
def make_collate_fn(pad_token_id):
    """
    Pad each batch to the longest sequence in that batch (more efficient than
    padding everything to a global max length in the dataset).
    """
    def collate_fn(batch):
        input_ids      = pad_sequence([b["input_ids"]      for b in batch], batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
        labels         = pad_sequence([b["labels"]         for b in batch], batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return collate_fn


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:          # Qwen has no pad token by default
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.to(device)

    # Data
    df = pd.read_json(DATA_PATH, lines=True)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=897401, shuffle=True)

    train_dataset = SarcasmDataset(train_df, tokenizer)
    val_dataset   = SarcasmDataset(val_df,   tokenizer)

    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Optimizer — weight decay on all params except biases and LayerNorm
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer    = AdamW(optimizer_grouped, lr=LR)
    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.amp.GradScaler("cuda", enabled=FP16)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with autocast('cuda', enabled=FP16):
                # AutoModelForCausalLM computes cross-entropy loss internally
                # when labels are passed, averaged over non-(-100) positions.
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if step % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * GRAD_ACCUM

            if step % 50 == 0:
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} "
                      f"| loss {train_loss / step:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                val_loss += outputs.loss.item()

        avg_val   = val_loss   / len(val_loader)
        avg_train = train_loss / len(train_loader)
        print(f"\nEpoch {epoch} complete | train loss: {avg_train:.4f} | val loss: {avg_val:.4f}\n")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(OUTPUT_DIR + "best")
            tokenizer.save_pretrained(OUTPUT_DIR + "best")
            print(f"  -> New best checkpoint saved (val loss {best_val_loss:.4f})\n")

    print("Training complete.")


# ── Inference ─────────────────────────────────────────────────────────────────
def generate(neutral_headline: str, checkpoint: str = OUTPUT_DIR + "best") -> str:
    """Load the best checkpoint and generate a sarcastic version of a headline."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model     = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Rewrite this headline: {neutral_headline}"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,   # appends <|im_start|>assistant\n
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LEN,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Slice off the prompt — decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()

    # Quick smoke test after training
    test_headline = "Local man wins award for community service"
    print("\nTest headline:", test_headline)
    print("Sarcastic version:", generate(test_headline))
