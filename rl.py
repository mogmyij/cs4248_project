import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from sentence_transformers import SentenceTransformer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from rewards.degeneration_reward import DegenerationReward
from rewards.content_reward import ContentReward
from rewards.config import RewardConfig
from rewards.diversity_reward import DiversityReward
from rewards.template_hack_reward import AreaManReward

MODEL_PATH = "./qwen_sarcasm_best" 
DATASET_PATH = "./dataset/rl_headlines_200k.jsonl"

SARCASM_MODEL_ID = "helinivan/english-sarcasm-detector"
CONTEXT_MODEL_ID = "all-mpnet-base-v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

debug_print_count = 0

NUM_GENERATIONS = 8

SARCASM_GATE_THRESHOLD = 0.7

EVAL_EVERY_STEPS = 100
EVAL_HEADLINES = [
        "CNA Explains: How the Iran war might reshape Asia’s energy playbook",
        "Iran downs two US warplanes as both sides race to find missing crew member",
        "OpenAI’s Top Executive Fidji Simo to Take Medical Leave From Company",
        "The New Jobs Being Created by AI",
        "Faced with new energy shock, Europe asks if reviving nuclear is the answer",
        "Artemis II crew now halfway to Moon as they take 'spectacular' image of Earth",
        "WHO warns about attacks on Iran health facilities, regional threat",
        "In a time of war, Chinese museums are a safe haven for ancient treasures of Iran",
        "Safety fears bloom in Japan as ageing cherry trees collapse in Tokyo parks",
        "Hong Kong airport opens sensory space for passengers with invisible disabilities",
        "What Happened to the Fun Parts of Work?",
        "Dvalishvili refuses surgery despite nose breaks",
        "What can F1's bosses do to help keep Verstappen in the sport?",
        "NCT’s Mark to leave K-pop group, SM Entertainment after 10 years",
        "Your hand feels tingly while using a charging phone? What causes it and is it still safe?",
        ]



# --- Initialize Reward Models Globally ---
# This prevents reloading them for every batch
print("Loading Reward Models...")
sarcasm_tokenizer = AutoTokenizer.from_pretrained(SARCASM_MODEL_ID)
sarcasm_model = AutoModelForSequenceClassification.from_pretrained(SARCASM_MODEL_ID).to(device)
sarcasm_model.eval()

context_model = SentenceTransformer(CONTEXT_MODEL_ID, device=device)

config = RewardConfig()
config.degen_repetition_weight = 0.7
config.degen_nonsense_weight = 0.3
config.ngram_sizes = [2,3]
config.special_char_threshold =  0.3
config.uppercase_threshold = 0.7
dg = DegenerationReward(config)
cr = ContentReward(config)
dr = DiversityReward(NUM_GENERATIONS)
amr = AreaManReward(config)


# --- Reward Functions ---
# GRPOTrainer expects functions that take 'completions' (list of generated strings)
# and kwargs containing dataset columns (like 'original_text').
def sarcasm_reward_func(completions, **kwargs):
    """Reward based on how sarcastic the generated headline is."""
    rewards = []
    clean_completions = []
    for c in completions:
        text = c.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        first_line = text.split('\n')[0].strip()
        clean_completions.append(first_line)

    with torch.no_grad():
        for i in range(0, len(clean_completions), 16):
            batch = clean_completions[i:i+16]
            inputs = sarcasm_tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits = sarcasm_model(**inputs).logits
            probs = F.softmax(logits, dim=1)

            # Probability of class 1 (Sarcastic)
            batch_rewards = probs[:, 1].cpu().tolist()
            rewards.extend(batch_rewards)

    #if (debug_print_count % 10 == 0 ):
    #    print("---------sarc rewards--------\n")
    #    print(f"reward: {rewards[0]}\n")
    #    print("------------------------------\n")


    torch.cuda.empty_cache()
    return rewards

def gated_context_reward_func(prompts, completions, original_text, **kwargs):
    global debug_print_count


    """Reward based on semantic similarity to the original non-sarcastic headline."""
    #original_headlines = kwargs.get("original_text", [])

    rewards = []
    clean_completions = []
    for c in completions:
        text = c.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        first_line = text.split('\n')[0].strip() # <--- THE MAGIC FIX
        clean_completions.append(first_line)

    #if not original_headlines or len(original_headlines) == 0:
    #    original_headlines = []
    #    for p in prompts:
    #        # Slices out the text between our instruction and the stop token
    #        extracted_text = p.split("Rewrite this headline: ")[-1].split("<|im_end|>")[0].strip()
    #        original_headlines.append(extracted_text)
    original_headlines = original_text


    with torch.no_grad():
        for i in range(0, len(clean_completions), 16):
            batch_comp = clean_completions[i:i+16]
            batch_orig = original_text[i:i+16]

            emb_comp = context_model.encode(batch_comp, convert_to_tensor=True, device=device)
            emb_orig = context_model.encode(batch_orig, convert_to_tensor=True, device=device)

            sim = F.cosine_similarity(emb_comp, emb_orig)
            batch_rewards = sim.cpu().tolist()
            rewards.extend(batch_rewards)

    #if (debug_print_count % 10 == 0 ):
    #    print("\n--- DEBUG: INPUT VS OUTPUT ---")
    #    print(f"ORIGINAL (Serious):  {original_headlines[0]}")
    #    print(f"GENERATED (Sarcasm): {clean_completions[0]}")
    #    print("---------context rewards--------\n")
    #    print(f"reward: {rewards[0]}\n")
    #    print("------------------------------\n")

    #debug_print_count += 1

    torch.cuda.empty_cache()

    """implement reward threshold"""
    sarc_score = sarcasm_reward_func(completions)
    for idx, score in enumerate(sarc_score):
        rewards[idx] = 0 if score < SARCASM_GATE_THRESHOLD else rewards[idx]

    return rewards

def gated_content_reward_func(completions, original_text, **kwargs):
    # check if it fulfils the sarcasm score threshold

    """Reward for content preservation via NLI entailment and entity overlap."""
    clean = [
        c.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip().split('\n')[0].strip()
        for c in completions
    ]

    rewards = cr.score(original_text, clean)

    """implement reward threshold"""
    sarc_score = sarcasm_reward_func(completions)
    for idx, score in enumerate(sarc_score):
        rewards[idx] = 0 if score < SARCASM_GATE_THRESHOLD else rewards[idx]

    return rewards

def degeneration_reward_func(completions, **kwargs):
    """Penalty for repetitive or nonsensical output (returned as negative reward)."""
    clean = [
        c.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip().split('\n')[0].strip()
        for c in completions
    ]
    scores = dg.score(clean)
    return [-s for s in scores]

def diversity_reward_func(completions, **kwargs):
    """Penalty for low intra-group diversity across GRPO generations (returned as negative reward)."""
    penalties = dr.score(completions)
    return [-p for p in penalties]


def area_man_reward_func(completions, **kwargs):
    """Penalty for shortcut-template outputs such as 'Area Man' (returned as negative reward)."""
    penalties = amr.score(completions)
    return [-p for p in penalties]


def build_dataset():
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    def format_data(sample):
        prompt = (
                "<|im_start|>system\n"
                "You are a headline writer for a satirical news publication like the Onion. "
                "Transform the given genuine news headline into a sarcastic version, styled like the Onion.<|im_end|>\n"
                "<|im_start|>user\n"
                f"Rewrite this headline: {sample['input']}<|im_end|>\n"
                "<|im_start|>assistant\n"
                )
        return {
                "prompt": prompt, 
                "original_text": sample['input'] 
                }

    dataset = dataset.map(format_data, batched=False)
    return dataset
class QualitativeEvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, headlines, eval_every_steps=EVAL_EVERY_STEPS):
        self.model = model
        self.tokenizer = tokenizer
        self.headlines = headlines
        self.eval_every_steps = eval_every_steps
        self.eval_data = []  # rows of [step, original, generated, sarcasm, context]

    def _build_prompt(self, headline):
        return (
                "<|im_start|>system\n"
                "You are a headline writer for a satirical news publication like the Onion. "
                "Transform the given genuine news headline into a sarcastic version, styled like the Onion.<|im_end|>\n"
                "<|im_start|>user\n"
                f"Rewrite this headline: {headline}<|im_end|>\n"
                "<|im_start|>assistant\n"
                )

    def on_step_end(self, args: GRPOConfig, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.eval_every_steps != 0:
            return

        self.model.eval()
        results = []

        prompts = []
        with torch.no_grad():
            for headline in self.headlines:
                prompt = self._build_prompt(headline)
                prompts.append(prompt)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        )
                generated = self.tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                        ).strip().split("\n")[0].strip()
                results.append([headline, generated])

        originals = [r[0] for r in results]
        generated_texts = [r[1] for r in results]
        sarcasm_scores = sarcasm_reward_func(generated_texts)
        context_scores = gated_context_reward_func(prompts, generated_texts, originals)
        content_scores = gated_content_reward_func(generated_texts, originals)

        print(f"\n{'='*60}")
        print(f"Qualitative Eval — Step {state.global_step}")
        print(f"{'='*60}")
        for (orig, gen), sarc, ctx in zip(results, sarcasm_scores, context_scores):
            print(f"  IN:  {orig}")
            print(f"  OUT: {gen}")
            print(f"  SCORES — sarcasm: {sarc:.3f}, context: {ctx:.3f}\n")

        for (orig, gen), sarc, ctx in zip(results, sarcasm_scores, context_scores):
            self.eval_data.append([state.global_step, orig, gen, sarc, ctx])

        self.model.train()

    def on_train_end(self, args: GRPOConfig, state: TrainerState, control: TrainerControl, **kwargs):
        if wandb.run is not None and self.eval_data:
            table = wandb.Table(columns=["Step", "Original", "Generated", "Sarcasm", "Context"], data=self.eval_data)
            wandb.log({"qualitative_eval": table})
        elif wandb.run is None:
            print("WARNING: wandb.run is None, table not logged")


# --- Main Training Loop ---
def main():
    print("Loading Dataset...")
    dataset = build_dataset()

    print("Configuring QLoRA...")
    # 4-bit Quantization
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            )

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            #quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="sdpa"
            )

    # PEFT Config for LoRA. GRPOTrainer automatically handles the reference model 
    peft_config = LoraConfig(
            r=1,
            lora_alpha=32,
            target_modules="all-linear",
            lora_dropout=0,
            task_type="CAUSAL_LM",
            )

    eval_callback = QualitativeEvalCallback(model, tokenizer, EVAL_HEADLINES)

    # GRPO specific parameters
    training_args = GRPOConfig(
            output_dir="./qwen_grpo_rl-ed",
            learning_rate=3e-5,
            per_device_train_batch_size=4,  # Number of distinct prompts per step
            gradient_accumulation_steps=4,
            num_generations=NUM_GENERATIONS,              # G in GRPO: Group size for relative advantage
            #max_prompt_length=128,
            max_completion_length=128,
            temperature=0.7,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=50,
            #max_steps=50000, 
            num_train_epochs = 1,
            beta = 0.01,
            #report_to="none",
            report_to="wandb",
            save_steps = 2500,
            save_total_limit = 2,
            run_name="qwen-grpo-sarcasm-rl",
            reward_weights=[1.0,1.0,1.0,1.0,1.0,1.0], 
            multi_objective_aggregation="normalize_then_sum",
            )

    print("Initializing GRPO Trainer...")
    trainer = GRPOTrainer(
            model=model,
            #reward_funcs=[combined_multiplier_func],
            reward_funcs=[sarcasm_reward_func, gated_context_reward_func, gated_content_reward_func, degeneration_reward_func, diversity_reward_func, area_man_reward_func],
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
            callbacks = [eval_callback],
            )

    print("Starting Training...")
    trainer.train()

    print("Saving the model...")
    trainer.save_model("./qwen_grpo_finetuned_final")
    print("Done!")

if __name__ == "__main__":
    main()
