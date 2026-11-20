# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""How much is a skill worth? One estimator, used for admission, re-tests and merges.

A skill is taken to a few check tasks. For every check task there is a baseline score: the mean
score of episodes on the same task that did not have the skill and were otherwise identical. For
every episode ``k`` that had the skill in its index::

    influence_k = mean over the tokens generated after the load of
                  log pi(token | context with the body) - log pi(token | context with the body blanked)

    u_k = 0                                        if the episode never loaded the skill
          clip(influence_k / tau, 0, 1)            if it did

    credit = sum_k u_k * (r_k - baseline[task(k)]) / (sum_k u_k + kappa)

``u_k`` is how much the episode really used the skill: having a skill in the index is not using
it, and neither is loading it and then acting as if it were not there. ``kappa`` shrinks the
estimate towards zero when few episodes used the skill; with no use at all the credit is zero.

Pure Python, no torch: unit-testable anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


def influence(
    logprobs_with: Sequence[float],
    logprobs_without: Sequence[float],
    generated: Sequence[int],
    segment_end: int,
    shift: int,
) -> Optional[float]:
    """Mean log-prob change of the generated tokens after a loaded skill body.

    ``logprobs_with[p]`` is the log-prob of token ``p`` of the real trajectory;
    ``logprobs_without[p + shift]`` is the log-prob of the same token when the body is blanked
    (the blanked message has a different length, hence the shift). ``generated[p]`` is 1 on tokens
    the policy produced. Only positions at or after ``segment_end`` (the end of the load
    observation) count: earlier tokens cannot depend on the body.

    Returns None when the episode generated nothing after the load.
    """
    total, n = 0.0, 0
    for p in range(segment_end, len(logprobs_with)):
        if generated[p]:
            total += logprobs_with[p] - logprobs_without[p + shift]
            n += 1
    return total / n if n else None


def usage_weight(loaded: bool, influence_value: Optional[float], tau: float) -> float:
    """``u_k`` of one episode. A load without a measurable influence (nothing generated after it) is 0."""
    if not loaded or influence_value is None:
        return 0.0
    if tau <= 0:
        return 1.0  # influence gating disabled: a load counts as full use
    return min(1.0, max(0.0, influence_value / tau))


@dataclass
class Sample:
    """One episode that had the skill under test in its index."""

    score: float
    baseline: float
    """Mean score of the episodes without the skill on the same check task."""
    weight: float
    """Usage weight ``u_k``."""
    loaded: bool = False


@dataclass
class Credit:
    value: float
    numerator: float
    """sum_k u_k * (r_k - baseline)"""
    weight: float
    """sum_k u_k: the effective number of episodes that used the skill."""
    n: int
    n_loaded: int

    def as_dict(self) -> dict[str, float]:
        return {
            "credit": round(self.value, 6),
            "numerator": round(self.numerator, 6),
            "weight": round(self.weight, 6),
            "n": self.n,
            "n_loaded": self.n_loaded,
        }


def credit(samples: Sequence[Sample], kappa: float) -> Credit:
    numerator = sum(s.weight * (s.score - s.baseline) for s in samples)
    weight = sum(s.weight for s in samples)
    return Credit(
        value=numerator / (weight + kappa) if weight + kappa > 0 else 0.0,
        numerator=numerator,
        weight=weight,
        n=len(samples),
        n_loaded=sum(int(s.loaded) for s in samples),
    )


def pooled_credit(parts: Sequence[tuple[float, float]], kappa: float) -> float:
    """Credit over the union of several sample sets, from their ``(numerator, weight)`` pairs."""
    numerator = sum(p[0] for p in parts)
    weight = sum(p[1] for p in parts)
    return numerator / (weight + kappa) if weight + kappa > 0 else 0.0


def group_normalize(values: Sequence[float], by_std: bool = True, epsilon: float = 1e-6) -> list[float]:
    """GRPO normalization of one comparison group: subtract the mean, optionally divide by the std.

    Uses the unbiased std like ``torch.std`` so the numbers match the trainer's solve groups. A
    group with fewer than two members carries no comparison and gets zeros.
    """
    n = len(values)
    if n < 2:
        return [0.0] * n
    mean = sum(values) / n
    centered = [v - mean for v in values]
    if not by_std:
        return centered
    std = math.sqrt(sum(c * c for c in centered) / (n - 1))
    return [c / (std + epsilon) for c in centered]


def write_advantages(
    credits_per_episode: Sequence[Sequence[Optional[float]]], by_std: bool = True
) -> list[list[float]]:
    """Group-normalized credit of every ``write_skill`` call of one source-task group.

    ``credits_per_episode[i]`` holds the credits of the skills episode ``i`` wrote (empty if it
    wrote none). An episode that wrote nothing enters the group as one credit of 0: not writing is
    worth exactly "no skill". That is what gives the *decision* to write a gradient: a call whose
    skill beats the group mean is reinforced, one whose skill falls below it is discouraged, and
    a group with a single writer still has a comparison.

    A credit of None marks a skill that was not tested (over the test budget): it gets advantage 0
    and is left out of the group.
    """
    flat, owner = [], []
    for i, credits in enumerate(credits_per_episode):
        if credits:
            for j, c in enumerate(credits):
                if c is None:
                    continue
                flat.append(float(c))
                owner.append((i, j))
        else:
            flat.append(0.0)
            owner.append((i, None))
    normalized = group_normalize(flat, by_std)
    out: list[list[float]] = [[0.0] * len(c) for c in credits_per_episode]
    for value, (i, j) in zip(normalized, owner, strict=True):
        if j is not None:
            out[i][j] = value
    return out
