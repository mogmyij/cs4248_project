"""
R_area_man(y): Template-hack penalty.

Penalises outputs that rely on the shortcut pattern "Area Man" rather than
learning broader sarcastic headline style. Higher score = more hacky = larger
penalty in the composite formula.

This reward is intentionally simple and rule-based so it is hard to exploit:
1. Exact lexical match for "area man"
2. Optional nearby variants such as "local man" or "Florida man"
3. Optional title-position boost, since headline hacks often appear at the start
"""

import re
from typing import List, Union

from .config import RewardConfig


class AreaManReward:
    """
    Rule-based penalty for Onion-style shortcut templates such as "Area Man".

    Parameters
    ----------
    config : RewardConfig
        Configuration object containing pattern list and scaling factors.
    """

    def __init__(self, config: RewardConfig):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in config.area_man_patterns]
        self.base_penalty = config.area_man_base_penalty
        self.prefix_boost = config.area_man_prefix_boost
        self.max_penalty = config.area_man_max_penalty
        self.prefix_window = config.area_man_prefix_window

    @staticmethod
    def _clean(text: str) -> str:
        text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        return text.split("\n")[0].strip()

    def _score_one(self, text: str) -> float:
        text = self._clean(text)
        lowered = text.lower()

        penalty = 0.0

        for pattern in self.patterns:
            match = pattern.search(lowered)
            if match is None:
                continue

            penalty += self.base_penalty

            if match.start() <= self.prefix_window:
                penalty += self.prefix_boost

        return min(penalty, self.max_penalty)

    def score(self, generated: Union[str, List[str]]) -> Union[float, List[float]]:
        """
        Compute template-hack score.

        Higher = more likely to be exploiting the "Area Man" shortcut and should
        be subtracted from total reward.
        """
        single = isinstance(generated, str)
        if single:
            generated = [generated]

        results = [self._score_one(text) for text in generated]
        return results[0] if single else results
