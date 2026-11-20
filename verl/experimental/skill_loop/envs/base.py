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
"""Environment contract of the skill loop.

The skill loop is benchmark-agnostic: it only needs a text environment whose outcome is verified
programmatically, plus the two prompt fragments that are benchmark specific (how to act, and what
a task of this benchmark looks like). ALFWorld is the reference implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskSpec:
    """One task instance, as stored in the dataset's ``extra_info``."""

    task_id: str
    """Stable id of the instance; used to keep a skill away from the task it was distilled from."""
    family: str
    """Task family (e.g. ALFWorld task type). Skills are expected to transfer within a family."""
    payload: dict[str, Any] = field(default_factory=dict)
    """Benchmark specific locator, e.g. ``{"gamefile": ...}``."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskSpec:
        return cls(task_id=str(d["task_id"]), family=str(d["family"]), payload=dict(d.get("payload") or {}))


@dataclass
class StepResult:
    observation: str
    done: bool
    won: bool
    valid: bool = True
    """False when the environment rejected the command (it still consumes a step)."""


class TextEnv(ABC):
    """A synchronous text environment. All methods are called from the worker's env thread."""

    max_steps: int

    @abstractmethod
    def reset(self) -> str:
        """Start the episode and return the first observation (including the task statement)."""

    @abstractmethod
    def step(self, command: str) -> StepResult: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


def episode_score(won: bool, steps: int, max_steps: int, efficiency_weight: float = 0.5) -> float:
    """Programmatic task score in [0, 1].

    Failure is 0. Success is worth ``1 - efficiency_weight * steps / max_steps``, i.e. between
    ``1 - efficiency_weight`` and 1. The efficiency term matters for skills: a skill that turns a
    25-step success into a 9-step success is useful, and a binary reward cannot see it.
    """
    if not won:
        return 0.0
    return 1.0 - efficiency_weight * min(steps, max_steps) / max_steps
