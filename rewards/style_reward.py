"""
  1. Length reward – ensures that sarcastic headline is not too long
  2. ROUGE-L score  – ensure that sarcastic headline is not too similar to headline
"""

from typing import List, Union
import math

from rouge_score import rouge_scorer


class StyleReward:
    def __init__(self, rouge_type: str = "rougeL", threshold = 2, steepness = 10):
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        self.threshold = threshold
        self.steepness = steepness


    def _clean(self, text: str) -> str:
        """Strip special tokens and take the first line."""
        text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        first_line = text.split("\n")[0].strip()
        return first_line

    def rouge_score(self, original, sarcastic) -> float:
        """
        Returns a value in [0, 1]:
          - 0.0 = all texts are completely different (maximally diverse)
          - 1.0 = all texts are identical (fully collapsed)
        """
        scores = self.scorer.score(original, sarcastic)
        return 1.0 - scores[self.rouge_type].fmeasure
 

    def length_score(self, original, sarcastic) -> float:
        gen_len = len(sarcastic)
        input_len = len(original)
        ratio = gen_len/input_len
        return 1 / (1 + math.exp(self.steepness * (ratio - self.threshold)))

 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        original: Union[str, List[str]],
        generated: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        cleaned = [self._clean(t) for t in generated]
        results = []
        for i in range(len(original)):
            results.append(self.rouge_score(original[i], cleaned[i]) + self.length_score(original[i], cleaned[i]))

        return results

