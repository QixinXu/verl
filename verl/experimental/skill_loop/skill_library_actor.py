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
"""Ray actor that owns the single skill library of a training run.

verl's rollout is spread over many ``AgentLoopWorker`` actors that share nothing. The library is
the one piece of state that must be shared across them *and* across training steps, so it is a
named actor: agent loop workers look it up to read/stage, the trainer looks it up to commit.
"""

from __future__ import annotations

from typing import Any, Optional

import ray

from verl.experimental.skill_loop.skill_library import SkillLibrary

SKILL_LIBRARY_ACTOR_NAME = "verl_skill_library"


class SkillLibraryActor:
    def __init__(self, config: Optional[dict[str, Any]] = None, load_path: Optional[str] = None):
        self.library = SkillLibrary(config)
        if load_path:
            self.library.load(load_path)

    def retrieve(
        self, query: str, task_id: str = "", n: Optional[int] = None, exclude_ids: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        return {"version": self.library.version, "skills": self.library.retrieve(query, task_id, n, exclude_ids)}

    def cluster_texts(self, texts: list[str], threshold: Optional[float] = None) -> list[int]:
        return self.library.cluster_texts(texts, threshold)

    def retest_plan(self, family: str, step: int, k: Optional[int] = None) -> list[dict[str, Any]]:
        return self.library.retest_plan(family, step, k)

    def merge_plan(self, family: str, step: int) -> Optional[list[dict[str, Any]]]:
        return self.library.merge_plan(family, step)

    def propose(self, proposal: dict[str, Any]) -> None:
        self.library.propose(proposal)

    def report_loads(self, loads: list[dict[str, Any]]) -> None:
        self.library.report_loads(loads)

    def report_retest(self, result: dict[str, Any]) -> None:
        self.library.report_retest(result)

    def report_merge(self, result: dict[str, Any]) -> None:
        self.library.report_merge(result)

    def commit(self, step: int) -> dict[str, Any]:
        return self.library.commit(step)

    def stats(self) -> dict[str, Any]:
        return self.library.stats()

    def state_dict(self) -> dict[str, Any]:
        return self.library.state_dict()

    def save(self, path: str) -> str:
        return self.library.save(path)

    def load(self, path: str) -> dict[str, Any]:
        self.library.load(path)
        return self.library.stats()


def create_skill_library_actor(config: Optional[dict[str, Any]] = None, load_path: Optional[str] = None):
    """Create (or attach to) the run's library actor. The caller must keep the handle alive."""
    return (
        ray.remote(SkillLibraryActor)
        .options(name=SKILL_LIBRARY_ACTOR_NAME, get_if_exists=True, num_cpus=0)
        .remote(config, load_path)
    )


def get_skill_library_actor():
    return ray.get_actor(SKILL_LIBRARY_ACTOR_NAME)
