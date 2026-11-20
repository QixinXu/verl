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
"""Rollout-only harness: the whole skill-evolution data path of the trainer, minus the model engine.

It runs the *same code* as ``SkillEvolveTrainerSync`` for everything between "a prompt leaves the
dataloader" and "a trajectory has an advantage": dataloader, TransferQueue, the Ray agent loop
workers, the replay buffer, role-grouped advantages and the per-step library commit. What it leaves
out is what needs GPUs: the actor update and the inference server. The LLM is injected as a client
object, so the harness can be driven by a scripted policy on a CPU node (CI, debugging) or by any
``LLMServerClient``.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import transfer_queue as tq

from verl.experimental.skill_loop.skill_trainer import SkillEvolveTrainerSync
from verl.utils.skip import SkipManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class SkillRolloutOnlyTrainer(SkillEvolveTrainerSync):
    """``SkillEvolveTrainerSync`` without worker groups. Requires ``rollout_correction.bypass_mode``."""

    def _setup(self):
        self._init_tokenizer()
        self._init_dataloader()
        self.global_steps = 0
        # rule rewards are computed inside the agent loop, so no reward workers are needed;
        # the non-None handle list only tells shared trainer code "rewards are already there".
        self.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[])
        self.validation_generations_logger = None
        SkipManager.init(self.config)

    def on_init_end(self):
        return

    def rollout_step(self) -> dict[str, Any]:
        """One training step without the gradient: rollout -> sample -> advantage -> library commit."""
        self.global_steps += 1
        SkipManager.set_step(self.global_steps)
        metrics: dict[str, Any] = {}
        self._add_batch_to_generate()
        batch, _ = self.replay_buffer.sample(
            global_steps=self.global_steps, partition_id="train", batch_size=self.config.data.train_batch_size
        )
        batch = self._compute_old_log_prob(batch, metrics)
        batch = self._compute_advantage(batch, metrics)
        metrics["rows"] = len(batch.keys)
        metrics["library"] = self._commit_skill_library()
        tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
        return metrics

    def validate(self) -> dict[str, float]:
        return self._validate()

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns, **kwargs):
        """Per-data-source means of every numeric reward info (the stock version also feeds loggers)."""
        metrics: dict[str, float] = {}
        sources = np.array(data_sources)
        for key, values in reward_extra_infos_dict.items():
            values = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
            for source in sorted(set(data_sources)):
                metrics[f"val/{source}/{key}"] = float(np.nanmean(values[sources == source]))
            metrics[f"val/all/{key}"] = float(np.nanmean(values))
        return metrics
