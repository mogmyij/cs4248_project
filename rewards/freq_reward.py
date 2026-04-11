from collections import Counter, deque
from typing import List

from .config import RewardConfig


class FrequencyPenalty:
    def __init__(self, config: RewardConfig):
        self.window_size = config.freq_window_size
        self.ngram_range = config.freq_ngram_range
        self.prefix_tokens = config.freq_prefix_token_count
        self.penalty_scale = config.freq_penalty_scale
        self.warmup = config.freq_penalty_warmup

        # Rolling window: each entry is a set of n-gram tuples for one generation
        self.window: deque = deque(maxlen=self.window_size)
        # Aggregate counts across the entire window (kept in sync)
        self.global_counts: Counter = Counter()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        """Strip special tokens and take the first line (same as other rewards)."""
        text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        return text.split("\n")[0].strip()

    def _extract_ngrams(self, text: str) -> set:
        """Extract n-grams from the opening tokens only."""
        tokens = text.lower().split()[: self.prefix_tokens]
        ngrams = set()
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.add(tuple(tokens[i : i + n]))
        return ngrams

    def _penalty_for(self, ngram_set: set) -> float:
        if not ngram_set or len(self.window) < self.warmup:
            return 0.0

        window_len = len(self.window)
        max_freq = 0.0
        for ng in ngram_set:
            count = self.global_counts.get(ng, 0)
            # Normalise: what fraction of past generations contained this n-gram?
            freq = count / window_len
            if freq > max_freq:
                max_freq = freq

        # Scale linearly up to penalty_scale, then clip
        return min(self.penalty_scale * max_freq, self.penalty_scale)

    def _update_window(self, batch_ngrams: List[set]) -> None:
        for ngram_set in batch_ngrams:
            # If window is at capacity, the deque auto-evicts the oldest
            # entry — we need to decrement its counts first.
            if len(self.window) == self.window_size:
                evicted = self.window[0]  # will be popped by deque
                for ng in evicted:
                    self.global_counts[ng] -= 1
                    if self.global_counts[ng] <= 0:
                        del self.global_counts[ng]

            self.window.append(ngram_set)
            for ng in ngram_set:
                self.global_counts[ng] += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, completions: List[str]) -> List[float]:
        cleaned = [self._clean(c) for c in completions]
        batch_ngrams = [self._extract_ngrams(t) for t in cleaned]

        # 1. Score FIRST (using counts accumulated BEFORE this batch)
        penalties = [self._penalty_for(ng_set) for ng_set in batch_ngrams]

        # 2. THEN update the window with this batch
        self._update_window(batch_ngrams)

        return penalties

    def reset(self) -> None:
        """Clear the rolling window (e.g. between training runs)."""
        self.window.clear()
        self.global_counts.clear()
