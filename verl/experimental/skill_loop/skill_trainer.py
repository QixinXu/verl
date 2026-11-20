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
"""Trainer side of skill-based self-evolution.

Two things differ from the stock synchronous V1 trainer, and nothing else:

1. ``_compute_advantage_estimates``: advantages are computed per ``(uid, arm)`` from each episode's
   own task score, and the tokens of ``write_skill`` calls additionally get the normalized credit
   of the skill they wrote (see ``skill_advantage.py``).
2. ``on_step_end``: after the policy update, the skill library commits the evidence staged by this
   step's rollouts. Committing here (and nowhere else) is what makes the library a per-step
   snapshot: all rollouts of step ``t`` read version ``t-1``, exactly like they read weights ``t-1``.

Enable with ``+trainer.v1.trainer_class=verl.experimental.skill_loop.skill_trainer.SkillEvolveTrainerSync``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np
import ray
import torch
import transfer_queue as tq

from verl import DataProto
from verl.experimental.skill_loop.skill_advantage import compute_skill_advantage, role_of
from verl.experimental.skill_loop.skill_library_actor import get_skill_library_actor
from verl.experimental.skill_loop.skill_loop_tq import SkillLoopConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _tolist(column: Any, n: int) -> list:
    if isinstance(column, str):  # a column whose rows are all equal may come back as one value
        return [column] * n
    return column.tolist() if hasattr(column, "tolist") else list(column)


def _padded_like(column: Any, like: torch.Tensor) -> torch.Tensor:
    """A token-level column (possibly a jagged nested tensor) as a dense tensor shaped like ``like``."""
    if getattr(column, "is_nested", False):
        column = torch.nested.to_padded_tensor(column, 0.0)
    out = torch.zeros(like.shape, dtype=torch.float32, device=like.device)
    width = min(column.shape[1], like.shape[1])
    out[:, :width] = column[:, :width].to(out.dtype)
    return out


class SkillEvolveTrainerSync(PPOTrainerSync):
    """Synchronous GRPO trainer with per-arm advantages, write-call credit and a per-step library commit."""

    @property
    def skill_cfg(self) -> SkillLoopConfig:
        if not hasattr(self, "_skill_cfg"):
            self._skill_cfg = SkillLoopConfig.from_config(self.config)
        return self._skill_cfg

    # ------------------------------------------------------------------ credit assignment

    def _compute_advantage_estimates(self, data: DataProto, batch, metrics: dict) -> DataProto:
        if not self.skill_cfg.enable:
            # the worker ran the stock per-session loop: rows carry no adv_group column
            return super()._compute_advantage_estimates(data, batch, metrics)
        if self.config.algorithm.adv_estimator != core_algos.AdvantageEstimator.GRPO:
            raise ValueError("skill_loop assigns credit with group baselines; set algorithm.adv_estimator=grpo")

        n = len(batch.keys)
        meta = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["adv_group", "skill_meta", "write_mask", "write_adv"],
        )
        adv_groups = [str(g) for g in _tolist(meta["adv_group"], n)]
        metas = [json.loads(m) for m in _tolist(meta["skill_meta"], n)]
        uids = [str(u) for u in data.non_tensor_batch["uid"]]
        scores = data.batch["token_level_rewards"].sum(dim=-1)
        response_mask = data.batch["response_mask"]
        write_mask = _padded_like(meta["write_mask"], response_mask)
        write_adv = _padded_like(meta["write_adv"], response_mask)

        advantages = compute_skill_advantage(
            scores=scores,
            uids=uids,
            adv_groups=adv_groups,
            response_mask=response_mask,
            write_mask=write_mask,
            write_adv=write_adv,
            norm_adv_by_std=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            role_coef=self.skill_cfg.role_adv_coef,
            write_task_coef=self.skill_cfg.write_task_coef,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = advantages

        is_padding = [bool(tag.get("is_padding", False)) for tag in batch.tags]
        # advantage of the ordinary (non write_skill) tokens, and of the write_skill tokens
        plain = response_mask * (1 - write_mask)
        solve_adv = (advantages * plain).sum(-1) / plain.sum(-1).clamp(min=1)
        wrote = write_mask.sum(-1)
        write_token_adv = (advantages * write_mask).sum(-1) / wrote.clamp(min=1)
        metrics.update(_skill_metrics(metas, uids, is_padding))
        self._dump_advantages(
            batch.keys, adv_groups, metas, solve_adv.tolist(), write_token_adv.tolist(), wrote.tolist(), is_padding
        )
        return data

    def _dump_advantages(self, keys, adv_groups, metas, solve_adv, write_adv, wrote, is_padding) -> None:
        dump_dir = self.skill_cfg.dump_dir
        if not dump_dir:
            return
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, "advantages.jsonl"), "a") as f:
            for key, group, meta, adv, w_adv, n_write, pad in zip(
                keys, adv_groups, metas, solve_adv, write_adv, wrote, is_padding, strict=True
            ):
                if pad:
                    continue
                f.write(
                    json.dumps(
                        {
                            "step": self.global_steps,
                            "key": key,
                            "role": role_of(group),
                            "adv_group": group,
                            "reward": round(float(meta["score"]), 6),
                            "advantage": round(float(adv), 6),
                            "write_tokens": int(n_write),
                            "write_advantage": round(float(w_adv), 6) if n_write else None,
                        }
                    )
                    + "\n"
                )

    # ------------------------------------------------------------------ library lifecycle

    def on_init_end(self):
        super().on_init_end()
        # 0 for a fresh run, N when _load_checkpoint resumed from global_step_N
        self._resumed_step = int(self.global_steps)

    def on_train_begin(self):
        super().on_train_begin()
        self._restore_skill_library()

    def on_validate_begin(self):
        super().on_validate_begin()
        self._restore_skill_library()  # validation before the first step must see the resumed library too

    def on_step_end(self):
        super().on_step_end()
        self._commit_skill_library()

    def _checkpoint_library_path(self, step: int) -> str:
        return os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}", "skill_library.json")

    def _restore_skill_library(self) -> None:
        """On resume, load the library that was committed together with the resumed weights.

        ``skill_evolve.library_path`` wins (the manager already loaded it). Otherwise a run resumed
        from ``global_step_N`` continues from ``global_step_N/skill_library.json``; without it the
        run would silently restart from an empty library.
        """
        if getattr(self, "_skill_library_restored", False):
            return
        self._skill_library_restored = True
        resumed_step = int(getattr(self, "_resumed_step", 0))
        if self.skill_cfg.library_path or resumed_step <= 0:
            return
        path = self._checkpoint_library_path(resumed_step)
        if os.path.exists(path):
            stats = ray.get(get_skill_library_actor().load.remote(path))
            logger.info(f"[skill_library] restored {path}: {stats}")
        else:
            logger.warning(f"[skill_library] resumed from step {resumed_step} but {path} is missing; library is empty")

    def _commit_skill_library(self) -> dict[str, Any]:
        """Apply this step's staged evidence to the library, log it and snapshot it to disk."""
        library = get_skill_library_actor()
        summary = ray.get(library.commit.remote(self.global_steps))
        events = summary.pop("events", [])
        per_family = summary.pop("per_family", {})
        logger.info(f"[skill_library] step {self.global_steps}: {summary} per_family={per_family} events={events}")

        pending = dict(getattr(self, "_pending_sync_metrics", None) or {})
        pending.update({f"skill_lib/{k}": v for k, v in summary.items() if k not in ("step",)})
        self._pending_sync_metrics = pending

        snapshot_dir = os.path.join(self.config.trainer.default_local_dir, "skill_library")
        ray.get(library.save.remote(os.path.join(snapshot_dir, f"step_{self.global_steps}.json")))
        ray.get(library.save.remote(os.path.join(snapshot_dir, "latest.json")))
        # keep the library that belongs to these weights next to them (the checkpoint of this step,
        # if any, was written just before this commit)
        if os.path.isdir(os.path.dirname(self._checkpoint_library_path(self.global_steps))):
            ray.get(library.save.remote(self._checkpoint_library_path(self.global_steps)))
        return {**summary, "per_family": per_family, "events": events}


def _corr(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _skill_metrics(metas: list[dict[str, Any]], uids: list[str], is_padding: list[bool]) -> dict[str, float]:
    """Per-role scores plus the quantities that tell whether skills are written, loaded, used and useful."""
    rows = [(m, u) for m, u, pad in zip(metas, uids, is_padding, strict=True) if not pad]
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m, _ in rows:
        by_role[role_of(m["arm"])].append(m)

    metrics: dict[str, float] = {}
    for role, ms in by_role.items():
        metrics[f"skill/{role}_score_mean"] = float(np.mean([m["score"] for m in ms]))
        metrics[f"skill/{role}_success"] = float(np.mean([m["won"] for m in ms]))
        metrics[f"skill/{role}_rows"] = float(len(ms))
        metrics[f"skill/{role}_load_frac"] = float(np.mean([m.get("n_loads", 0) > 0 for m in ms]))

    source = by_role.get("source", [])
    if source:
        metrics["skill/write_frac"] = float(np.mean([m["n_writes"] > 0 for m in source]))
        credits = [c for m in source for c in m.get("write_credits", [])]
        if credits:
            metrics["skill/write_credit_mean"] = float(np.mean(credits))
            metrics["skill/write_credit_pos_frac"] = float(np.mean([c > 0 for c in credits]))

    # the new skills on the check tasks: were they loaded, were they used, did the score move
    candidate = by_role.get("candidate", [])
    if candidate:
        usage = [m["usage"][m["subject"]] for m in candidate if m.get("usage", {}).get(m.get("subject"))]
        loaded = [u for u in usage if u["loaded"]]
        metrics["skill/candidate_loaded_frac"] = float(np.mean([u["loaded"] for u in usage])) if usage else 0.0
        if loaded:
            metrics["skill/candidate_usage_weight"] = float(np.mean([u["weight"] for u in loaded]))
            influence = [u["influence"] for u in loaded if u["influence"] is not None]
            if influence:
                metrics["skill/candidate_influence"] = float(np.mean(influence))
        # diagnostics of the usage weight: it should not simply track episode length or outcome
        measured = [m for m in candidate if m["usage"].get(m["subject"], {}).get("loaded")]
        weights = [m["usage"][m["subject"]]["weight"] for m in measured]
        metrics["skill/usage_weight_corr_steps"] = _corr(weights, [m["steps"] for m in measured])
        metrics["skill/usage_weight_corr_score"] = _corr(weights, [m["score"] for m in measured])

        # plain (intention-to-treat) lift of adding a new skill to the index, per (unit, check task)
        control: dict[tuple[str, int], list[float]] = defaultdict(list)
        treated: dict[tuple[str, int], list[float]] = defaultdict(list)
        for m, uid in rows:
            role = role_of(m["arm"])
            if role == "control":
                control[(uid, m["check"])].append(m["score"])
            elif role == "candidate":
                treated[(uid, m["check"])].append(m["score"])
        lifts = [np.mean(treated[k]) - np.mean(control[k]) for k in treated if k in control]
        if lifts:
            metrics["skill/candidate_lift"] = float(np.mean(lifts))
    return {k: v for k, v in metrics.items() if not (isinstance(v, float) and np.isnan(v))}
