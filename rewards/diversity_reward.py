"""
R_diversity(Y): Batch-level diversity penalty.
 
Measures how similar generations are to each other within a GRPO group.
High similarity across the group means the model has collapsed to a single
strategy (e.g. "Area Man" + copy-paste) — this penalises that.
 
Uses pairwise ROUGE-L F1 across all generations in the group.
A perfectly diverse group scores 0.0 (no penalty).
A fully collapsed group (identical outputs) scores 1.0 (max penalty).
 
Can be used alongside the per-sentence repetition check from
DegenerationReward, which catches within-sentence degeneration.
 
Usage with GRPOTrainer
----------------------
GRPO generates `num_generations` completions per prompt. This reward
function expects to receive ALL completions for a batch and the
`num_generations` parameter so it can split them into groups and score
each group independently.
 
    diversity = DiversityReward(num_generations=4)
 
    # Inside your reward function passed to GRPOTrainer:
    def diversity_penalty_func(completions, **kwargs):
        return diversity.score(completions)
        # Returns a list of per-sample penalties (one per completion)
"""
 
from itertools import combinations
from typing import List, Union
 
from rouge_score import rouge_scorer
 
 
class DiversityReward:
    """
    Pairwise ROUGE-L diversity penalty for GRPO generation groups.
 
    Parameters
    ----------
    num_generations : int
        Number of completions generated per prompt (the G in GRPO).
        Must match GRPOConfig.num_generations.
    rouge_type : str
        Which ROUGE variant to use. 'rougeL' is recommended because it
        captures longest common subsequence (structural similarity)
        rather than just unigram/bigram overlap.
    """
 
    def __init__(self, num_generations: int = 4, rouge_type: str = "rougeL"):
        self.num_generations = num_generations
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
 
    def _clean(self, text: str) -> str:
        """Strip special tokens and take the first line."""
        text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        first_line = text.split("\n")[0].strip()
        return first_line
 
    def _pairwise_rouge(self, texts: List[str]) -> float:
        """
        Average pairwise ROUGE-L F1 within a group of texts.
 
        Returns a value in [0, 1]:
          - 0.0 = all texts are completely different (maximally diverse)
          - 1.0 = all texts are identical (fully collapsed)
        """
        if len(texts) <= 1:
            return 0.0
 
        pairs = list(combinations(range(len(texts)), 2))
        if not pairs:
            return 0.0
 
        total = 0.0
        for i, j in pairs:
            result = self.scorer.score(texts[i], texts[j])
            total += result[self.rouge_type].fmeasure
 
        return total / len(pairs)
 
    def score(self, generated: List[str]) -> List[float]:
        """
        Compute per-sample diversity penalty.
 
        Parameters
        ----------
        generated : list of str
            All completions in the batch, ordered as GRPO produces them:
            [prompt0_gen0, prompt0_gen1, ..., prompt0_genG,
             prompt1_gen0, prompt1_gen1, ..., prompt1_genG, ...]
 
        Returns
        -------
        list of float
            One penalty score per completion. All completions in the same
            group get the same penalty (since diversity is a group property).
            Score in [0, 1] — higher = less diverse = bigger penalty.
        """
        cleaned = [self._clean(t) for t in generated]
        n = self.num_generations
        total = len(cleaned)
 
        if total % n != 0:
            raise ValueError(
                f"Batch size ({total}) is not divisible by "
                f"num_generations ({n}). Check that you're passing "
                f"all completions from the batch."
            )
 
        num_prompts = total // n
        penalties = []
 
        for p in range(num_prompts):
            group = cleaned[p * n : (p + 1) * n]
            group_penalty = self._pairwise_rouge(group)
            # Assign the same group-level penalty to every member
            penalties.extend([group_penalty] * n)
 
        return penalties
 
 
# --------------------------------------------------------------------------
# Standalone usage example / quick test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate a batch with num_generations=4, 2 prompts
    # Prompt 1: collapsed (all similar) — should get HIGH penalty
    # Prompt 2: diverse — should get LOW penalty
    test_completions = [
        # Prompt 1 group — collapsed
        "Area Man Refuses Surgery Despite Nose Breaks",
        "Area Man Refuses Surgery Despite Broken Nose",
        "Area Man Refuses Surgery Despite Nose Breaks",
        "Area Man Refuses Surgery Despite Breaking Nose",
        # Prompt 2 group — diverse
        "Local Fighter Tells Doctors To Take A Hike After Nose Shatters",
        "Nation's Bravest Idiot Declines Medical Attention For Mangled Face",
        "Man With Cartilage Where Bone Should Be Says He's Fine, Actually",
        "UFC Champion Discovers Bold New Alternative To Healthcare: Denial",
    ]
 
    div = DiversityReward(num_generations=4)
    scores = div.score(test_completions)
 
    print("Diversity penalties:")
    for i, (text, score) in enumerate(zip(test_completions, scores)):
        group = "COLLAPSED" if i < 4 else "DIVERSE"
        print(f"  [{group}] {score:.3f}  |  {text}")
 
