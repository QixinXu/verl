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
"""GRPO advantage for skill-evolution rollouts.

Stock GRPO compares the ``n`` sessions of one prompt on one scalar. A skill-evolution unit holds
several *arms* (see ``skill_loop_tq.py``); an episode is compared only with the episodes of the
same ``(uid, arm)``, which share its task and its skill index:

    A_i = coef[role] * (r_i - mean(arm)) / (std(arm) + eps)            on every generated token

The tokens of a ``write_skill`` call are the exception. Their reward is the task score *and* the
credit of the skill they wrote, each normalized in the source group and then added::

    A_token = write_task_coef * A_i + write_adv_token                  on write_skill tokens

where ``write_adv`` (the normalized credit, already scaled by ``write_credit_coef``) was computed
by the rollout worker, which is the only place that knows the credits of a whole group.

An arm with a single episode carries no comparison and gets zero advantage.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import torch

ROLES = ("source", "control", "candidate", "retest", "merge", "eval")


def role_of(adv_group: str) -> str:
    """``source`` -> source, ``check0/control`` -> control, ``check1/new2`` -> candidate, ..."""
    leaf = adv_group.split("/", 1)[1] if "/" in adv_group else adv_group
    for role in ROLES:
        if leaf.startswith(role):
            return role
    return "candidate" if "/" in adv_group else adv_group


def compute_skill_advantage(
    scores: torch.Tensor,
    uids: Sequence[str],
    adv_groups: Sequence[str],
    response_mask: torch.Tensor,
    write_mask: Optional[torch.Tensor] = None,
    write_adv: Optional[torch.Tensor] = None,
    norm_adv_by_std: bool = True,
    role_coef: Optional[dict[str, float]] = None,
    write_task_coef: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return token-level advantages of shape ``response_mask.shape``.

    Args:
        scores: ``(bs,)`` task score of each episode.
        uids: prompt uid of each episode.
        adv_groups: arm of each episode within its prompt.
        response_mask: ``(bs, response_length)``, 1 on model-generated tokens.
        write_mask: same shape, 1 on the tokens of ``write_skill`` calls.
        write_adv: same shape, normalized credit on those tokens.
        norm_adv_by_std: divide by the group std (GRPO) or not (Dr.GRPO).
        role_coef: optional per-role multiplier, e.g. ``{"control": 0.5}``.
        write_task_coef: weight of the task-score advantage on write_skill tokens.
    """
    role_coef = role_coef or {}
    members: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (uid, group) in enumerate(zip(uids, adv_groups, strict=True)):
        members[(uid, group)].append(i)

    advantages = torch.zeros_like(scores, dtype=torch.float32)
    with torch.no_grad():
        for (_, group), idx in members.items():
            if len(idx) < 2:
                continue
            group_scores = scores[idx].float()
            centered = group_scores - group_scores.mean()
            if norm_adv_by_std:
                centered = centered / (group_scores.std() + epsilon)
            advantages[idx] = centered * float(role_coef.get(role_of(group), 1.0))
        token_adv = advantages.unsqueeze(-1) * response_mask
        if write_mask is not None and write_adv is not None:
            write_mask = write_mask.to(token_adv.dtype) * response_mask
            token_adv = (
                token_adv * (1.0 - (1.0 - write_task_coef) * write_mask) + write_adv.to(token_adv.dtype) * write_mask
            )
    return token_adv
