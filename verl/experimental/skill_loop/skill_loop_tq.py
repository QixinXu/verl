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
"""Group-level rollout of skill-based self-evolution on top of verl's V1 TransferQueue worker.

Stock verl runs the ``n`` sessions of a prompt as independent agent loops and gives each session
one reward. Here one prompt is a *training unit* ``(A; B_1..B_m)``: tasks of one task type, all
different. The unit is run as a set of **arms**; an arm is a number of episodes on one task with
one skill index, and it is one GRPO comparison group::

    arm                       task  index in the prompt             episodes     purpose
    ------------------------  ----  ------------------------------  -----------  ---------------------------
    source                    A     retrieved(A)                    rollout.n    task score; write_skill
    check<j>/control          B_j   retrieved(B_j)                  control_n    baseline of B_j
    check<j>/<candidate g>    B_j   retrieved(B_j) + g              check_n      credit of the new skill g
    check<j>/retest_<s>/...   B_j   retrieved(B_j) with / without s retest_n     re-test of library skill s
    check<j>/merge/...        B_j   without / members / merged      retest_n     is a merged skill as good

Every episode is rewarded with its own task score and compared inside its arm: all episodes of an
arm have the same task *and* the same index, so nothing but their own actions (including which
skill they load) can explain a difference.

The credit of a skill (``credit.py``) is computed from the arms that carry it, against the mean
score of the arm that does not. It is the reward of the ``write_skill`` call that produced the
skill, it decides admission, and the same estimator re-tests library skills and judges merges.

``write_skill`` is offered in every episode, but only the skills written in source episodes are
tested; a write elsewhere is an ordinary action (it costs a step) and its skill is dropped.
A ``write_skill`` call is a few tokens inside a source episode. Its advantage is passed to the
trainer as two token-level columns next to ``response_mask``: ``write_mask`` (1 on the tokens of a
write call) and ``write_adv`` (the group-normalized credit on those tokens).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import hydra
import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    DictConfigWrap,
    ToolListWrap,
    _agent_loop_registry,
)
from verl.experimental.skill_loop import credit, prompts
from verl.experimental.skill_loop.envs.base import TaskSpec
from verl.experimental.skill_loop.skill_agent_loop import SkillAgentLoop, SolveResult
from verl.experimental.skill_loop.skill_library_actor import create_skill_library_actor, get_skill_library_actor
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQBase
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

SKILL_AGENT_NAME = "skill_agent"
MERGE_MAX_TOKENS = 320  # token cap of a drafted merged skill
# Re-test and merge arms only measure library skills; their episodes are recorded but not trained on.
MEASUREMENT_ONLY = ("retest_", "merge_")


@dataclass
class SkillLoopConfig:
    """``config.skill_evolve``. See ``verl/trainer/config/skill_evolve/default.yaml`` for the docs."""

    enable: bool = True
    num_checks: int = 2
    control_n: int = 4
    check_n: int = 4
    max_candidates: int = 4
    retest_n: int = 4
    tau: float = 0.1
    kappa: float = 1.0
    write_credit_coef: float = 1.0
    write_task_coef: float = 1.0
    val_use_skills: bool = True
    max_skill_words: int = 120
    dump_dir: Optional[str] = None
    library_path: Optional[str] = None
    agent: dict[str, Any] = field(default_factory=dict)
    library: dict[str, Any] = field(default_factory=dict)
    role_adv_coef: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: DictConfig) -> SkillLoopConfig:
        node = config.get("skill_evolve") if config is not None else None
        raw = OmegaConf.to_container(node, resolve=True) if node is not None else {}
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown skill_evolve config keys: {sorted(unknown)}")
        return cls(**raw)


@dataclass
class _Arm:
    """Episodes on one task with one skill index: one GRPO comparison group."""

    name: str
    role: str
    """source, control, candidate, retest_with, retest_without, merge_without, merge_members, merge_merged or eval."""
    task: TaskSpec
    index: list[dict[str, Any]]
    n: int
    measure: tuple[str, ...] = ()
    """Skill ids whose influence is measured in the episodes that load them."""
    check: int = -1
    """Position of the check task this arm runs on (-1 for the source task)."""
    subject: str = ""
    """Id of the skill this arm exists for (a candidate, or a library skill under re-test)."""
    results: list[SolveResult] = field(default_factory=list)

    def mean_score(self) -> float:
        return _mean([r.score for r in self.results])

    def record(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "check": self.check,
            "subject": self.subject,
            "task_id": self.task.task_id,
            "goal": self.task.payload.get("goal", ""),
            "index": [{k: e.get(k) for k in ("handle", "skill_id", "name", "when_to_use")} for e in self.index],
            "mean_score": self.mean_score(),
            "episodes": episodes,
        }


@dataclass
class _Row:
    """One episode on its way to TransferQueue."""

    session_id: int
    adv_group: str
    output: AgentLoopOutput
    reward: float
    info: dict[str, Any]
    """Scalars reported per data source by verl's validation (``reward_extra_info``)."""
    meta: dict[str, Any]
    """What the trainer needs to compute the step metrics (``skill_meta`` column)."""
    write_spans: list[tuple[int, int, float]] = field(default_factory=list)
    """``(start, end, advantage)`` of every write_skill call, in response coordinates."""


@dataclass
class _Tests:
    """Everything one unit puts to the test on its check tasks."""

    candidates: list[dict[str, Any]]
    """New skills written on the source task (de-duplicated)."""
    retests: list[dict[str, Any]]
    """Library skills due for a re-test."""
    cluster: list[dict[str, Any]]
    """Similar library skills to be merged (empty: no merge this step)."""
    merged: Optional[dict[str, Any]]
    """The drafted replacement of ``cluster`` (None: no draft, or it did not parse)."""

    @property
    def member_ids(self) -> list[str]:
        return [s["skill_id"] for s in self.cluster]


class SkillLoopWorkerTQ(AgentLoopWorkerTQBase):
    """TransferQueue worker that runs a prompt as a skill-evolution training unit."""

    def __init__(self, *args, **kwargs):
        # The loop must be registered before the base class checks the tokenizer against it.
        config = args[0] if args else kwargs["config"]
        self.skill_cfg = SkillLoopConfig.from_config(config)
        _agent_loop_registry[SKILL_AGENT_NAME] = {
            "_target_": f"{SkillAgentLoop.__module__}.{SkillAgentLoop.__qualname__}",
            "max_skill_words": self.skill_cfg.max_skill_words,
            **self.skill_cfg.agent,
        }
        super().__init__(*args, **kwargs)
        self._library = None
        self._dump_fh = None

    # ------------------------------------------------------------------ plumbing

    @property
    def library(self):
        if self._library is None:
            self._library = get_skill_library_actor()
        return self._library

    def _new_agent_loop(self) -> SkillAgentLoop:
        # same construction as AgentLoopWorker._run_agent_loop, but the worker keeps the instance
        # because it drives the arms itself.
        return hydra.utils.instantiate(
            config=_agent_loop_registry[SKILL_AGENT_NAME],
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.llm_client,
            tokenizer=self.tokenizer,
            processor=self.processor,
            hf_model_type=self.hf_model_type,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
            tools=ToolListWrap(self.tools),
        )

    async def _retrieve(self, task: TaskSpec) -> tuple[list[dict[str, Any]], int]:
        """Coarse recall of library skills for a task (the query is the task description)."""
        result = await self.library.retrieve.remote(query=task.payload.get("goal", ""), task_id=task.task_id)
        return result["skills"], result["version"]

    async def _run_arms(self, agent: SkillAgentLoop, arms: list[_Arm], sampling_params: dict) -> None:
        """Run every episode of ``arms`` concurrently."""
        owners = [arm for arm in arms for _ in range(arm.n)]
        results = await _gather_settled(
            [agent.solve(arm.task, sampling_params, index=arm.index, measure=arm.measure) for arm in owners]
        )
        for arm, result in zip(owners, results, strict=True):
            arm.results.append(result)

    # ------------------------------------------------------------------ entry point

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        if not self.skill_cfg.enable or prompt.get("agent_name") != SKILL_AGENT_NAME:
            return await super()._run_prompt(prompt, sampling_params, trajectory, trace)

        validate = trajectory["validate"]
        uid, partition_id = prompt["uid"], "train" if not validate else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        status = "failure"
        try:
            rollout_cfg = self.config.actor_rollout_ref.rollout
            n = int(prompt.pop("__rollout_n__", rollout_cfg.n if not validate else rollout_cfg.val_kwargs.n))
            prompt.pop("__do_sample__", None)
            if validate:
                rows, record = await self._run_eval_group(prompt, sampling_params, n)
            else:
                rows, record = await self._run_train_unit(prompt, sampling_params, n)
            # One batched put: a unit is either entirely in TransferQueue or not at all.
            await self._put_rows(prompt, rows, validate)
            self._dump(record, trajectory)
            status = "finished"
        except Exception as e:
            logger.exception(f"Error in skill loop _run_prompt for uid={uid}: {e}")
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})

    # ------------------------------------------------------------------ training unit

    async def _run_train_unit(self, prompt: dict, sampling_params: dict, n: int) -> tuple[list[_Row], dict]:
        cfg = self.skill_cfg
        extra_info = prompt["extra_info"]
        source = TaskSpec.from_dict(extra_info["source"])
        checks = [TaskSpec.from_dict(p) for p in list(extra_info.get("probes") or [])][: cfg.num_checks]
        step, uid = int(prompt.get("global_steps", 0)), prompt["uid"]
        agent = self._new_agent_loop()

        source_index, version = await self._retrieve(source)
        check_index = [(await self._retrieve(check))[0] for check in checks]

        # 1. the source task A: solve it with the retrieved index; any episode may write skills
        source_arm = _Arm("source", "source", source, _with_handles(source_index), n)
        await self._run_arms(agent, [source_arm], sampling_params)

        # 2. the check tasks B_j: new skills, library re-tests and at most one merge, all at once
        tests = _Tests(candidates=[], retests=[], cluster=[], merged=None)
        if checks:
            tests.candidates = await self._candidates(source_arm)
            tests.retests = await self.library.retest_plan.remote(source.family, step)
            tests.cluster = await self.library.merge_plan.remote(source.family, step) or []
            if tests.cluster:
                tests.merged = await self._draft_merge(agent, tests.cluster, sampling_params)
        arms = self._check_arms(uid, checks, check_index, tests)
        await self._run_arms(agent, arms, sampling_params)

        # 3. credit: of every new skill, of the re-tested skills, of the merge
        baseline = {a.check: a.mean_score() for a in arms if a.role == "control"}
        for cand in tests.candidates:
            cand["credit"] = self._credit(_select(arms, "candidate", cand["skill_id"]), (cand["skill_id"],), baseline)
        retests = [self._retest(arms, s, baseline, step) for s in tests.retests]
        merge = self._merge(arms, tests, source.family, step) if tests.cluster else None

        # 4. the write_skill calls are rewarded with the credit of their skill, normalized in the source group
        by_writer = {(i, k): cand for cand in tests.candidates for (i, k) in cand["writers"]}
        credits = [
            [by_writer[(i, k)]["credit"].value if (i, k) in by_writer else None for k in range(len(r.writes))]
            for i, r in enumerate(source_arm.results)
        ]
        by_std = bool(self.config.algorithm.get("norm_adv_by_std_in_grpo", True))
        write_adv = credit.write_advantages(credits, by_std)

        # 5. evidence for the library (applied at the trainer's step-end commit)
        candidates = [
            {
                **{k: c[k] for k in ("skill_id", "name", "when_to_use", "procedure", "writers", "duplicates")},
                **c["credit"].as_dict(),
            }
            for c in tests.candidates
        ]
        await self.library.report_loads.remote(_load_report([source_arm, *_select(arms, "control")]))
        for result in retests:
            await self.library.report_retest.remote(result)
        if merge:
            await self.library.report_merge.remote(merge)
        if candidates:
            await self.library.propose.remote(
                {
                    "uid": uid,
                    "family": source.family,
                    "source_task_id": source.task_id,
                    "step": step,
                    "candidates": candidates,
                }
            )

        # rows for TransferQueue and the human-readable record
        rows: list[_Row] = []
        arm_records = []
        for arm in [source_arm, *arms]:
            episodes = []
            for i, result in enumerate(arm.results):
                spans, writes = [], []
                if arm.role == "source":
                    for k, written in enumerate(result.writes):
                        advantage = cfg.write_credit_coef * write_adv[i][k]
                        spans.append((*written.span, advantage))
                        writes.append(
                            {
                                "skill": written.skill,
                                "step": written.step,
                                "candidate": by_writer[(i, k)]["skill_id"] if (i, k) in by_writer else None,
                                "credit": credits[i][k],
                                "write_advantage": round(advantage, 6),
                            }
                        )
                usage = self._usage(result, arm.measure)
                meta = {
                    **_episode_meta(arm, result),
                    "check": arm.check,
                    "subject": arm.subject,
                    "n_writes": len(result.writes),
                    "write_credits": [w["credit"] for w in writes if w["credit"] is not None],
                    "usage": usage,
                }
                if arm.role.startswith(MEASUREMENT_ONLY):
                    episodes.append(_episode_record(None, result, usage, writes))
                    continue
                info = {"success": float(result.won), "steps": result.steps}
                episodes.append(_episode_record(len(rows), result, usage, writes))
                rows.append(_Row(len(rows), arm.name, result.output, result.score, info, meta, spans))
            arm_records.append(arm.record(episodes))

        record = {
            "uid": uid,
            "kind": "train_unit",
            "family": source.family,
            "library_version": version,
            "source": _task_record(source),
            "checks": [_task_record(c) for c in checks],
            "baseline": {str(k): v for k, v in baseline.items()},
            "candidates": candidates,
            "retests": retests,
            "merge": merge,
            "arms": arm_records,
        }
        return rows, record

    def _check_arms(
        self, uid: str, checks: list[TaskSpec], check_index: list[list[dict[str, Any]]], tests: _Tests
    ) -> list[_Arm]:
        """The arms of a unit on its check tasks."""
        cfg = self.skill_cfg
        arms: list[_Arm] = []
        for j, check in enumerate(checks):
            old = check_index[j]
            old_ids = {e["skill_id"] for e in old}

            def arm(name, role, n, index, extra=(), measure=(), subject="", j=j, check=check) -> _Arm:
                # `extra` skills are put straight into the index, at a position seeded per arm
                index = _with_handles(index, extra, seed=f"{uid}|{j}|{name}")
                return _Arm(f"check{j}/{name}", role, check, index, n, measure=tuple(measure), check=j, subject=subject)

            # Control: the old library only. Its mean score is the baseline of this check task.
            # Influence is measured for every library skill a control episode loads: a re-tested
            # skill needs it, and for the others it is recorded as a diagnostic.
            arms.append(arm("control", "control", cfg.control_n, old, measure=sorted(old_ids)))

            # Experiment: old library + ONE new skill. It is not retrieved but put straight into the
            # index, so a retrieval miss is never blamed on the writer. Loading it is up to the policy.
            for cand in tests.candidates:
                sid = cand["skill_id"]
                arms.append(arm(sid, "candidate", cfg.check_n, old, extra=[cand], measure=[sid], subject=sid))

            # Re-test of a library skill s: "library with s" against "library without s". One of
            # the two is the control arm, so only the other one has to be run.
            for s in tests.retests:
                sid = s["skill_id"]
                if sid in old_ids:
                    rest = [e for e in old if e["skill_id"] != sid]
                    arms.append(arm(f"retest_{sid}/without", "retest_without", cfg.retest_n, rest, subject=sid))
                else:
                    arms.append(
                        arm(
                            f"retest_{sid}/with",
                            "retest_with",
                            cfg.retest_n,
                            old,
                            extra=[s],
                            measure=[sid],
                            subject=sid,
                        )
                    )

            # Merge: is ONE merged skill at least as good as the cluster it would replace?
            if tests.cluster:
                rest = [e for e in old if e["skill_id"] not in tests.member_ids]
                arms.append(arm("merge/without", "merge_without", cfg.retest_n, rest))
                arms.append(arm("merge/members", "merge_members", cfg.retest_n, rest, tests.cluster, tests.member_ids))
                if tests.merged:
                    merged_id = tests.merged["skill_id"]
                    arms.append(arm("merge/merged", "merge_merged", cfg.retest_n, rest, [tests.merged], [merged_id]))
        return arms

    async def _candidates(self, source_arm: _Arm) -> list[dict[str, Any]]:
        """New skills of a source group, de-duplicated. Duplicates share one candidate and its credit."""
        written = [(i, k, w.skill) for i, r in enumerate(source_arm.results) for k, w in enumerate(r.writes)]
        if not written:
            return []
        texts = [f"{skill['when_to_use']}\n{skill['procedure']}" for _, _, skill in written]
        representative = await self.library.cluster_texts.remote(texts)
        candidates: dict[int, dict[str, Any]] = {}
        for pos, (i, k, _) in enumerate(written):
            rep = representative[pos]
            if rep not in candidates:
                if len(candidates) >= self.skill_cfg.max_candidates:
                    continue  # over the test budget: this write gets no credit signal this step
                skill = written[rep][2]
                candidates[rep] = {
                    "skill_id": f"new{len(candidates)}",
                    **skill,
                    "text": prompts.render_skill(skill),
                    "writers": [],
                    "duplicates": 0,
                }
            candidates[rep]["writers"].append((i, k))
            candidates[rep]["duplicates"] += int(pos != rep)
        return list(candidates.values())

    async def _draft_merge(self, agent: SkillAgentLoop, cluster: list[dict], sampling_params: dict) -> Optional[dict]:
        """Ask the policy for one skill replacing ``cluster``. These tokens are not trained."""
        max_words = self.skill_cfg.max_skill_words
        text = await agent.complete(prompts.build_merge_messages(cluster, max_words), sampling_params, MERGE_MAX_TOKENS)
        skill = prompts.parse_skill(text, max_words)
        if skill is None:
            return None
        return {"skill_id": "merged", **skill, "text": prompts.render_skill(skill)}

    # ------------------------------------------------------------------ credit

    def _usage(self, result: SolveResult, skill_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        """Per measured skill: was it loaded, its influence and the resulting usage weight."""
        usage = {}
        for sid in skill_ids:
            loaded, value = result.loaded(sid), result.influence.get(sid)
            usage[sid] = {
                "loaded": loaded,
                "influence": None if value is None else round(value, 6),
                "weight": round(credit.usage_weight(loaded, value, self.skill_cfg.tau), 6),
            }
        return usage

    def _credit(self, arms: list[_Arm], skill_ids: tuple[str, ...], baseline: dict[int, float]) -> credit.Credit:
        """Credit of ``skill_ids`` (one skill, or a cluster counted as one) from the arms that carry them.

        ``baseline[j]`` is the mean score on check task ``j`` of the arm without the skill.
        """
        samples = []
        for arm in arms:
            for result in arm.results:
                usage = self._usage(result, skill_ids).values()
                samples.append(
                    credit.Sample(
                        score=result.score,
                        baseline=baseline[arm.check],
                        weight=max((u["weight"] for u in usage), default=0.0),
                        loaded=any(u["loaded"] for u in usage),
                    )
                )
        return credit.credit(samples, self.skill_cfg.kappa)

    def _retest(self, arms: list[_Arm], skill: dict[str, Any], baseline: dict[int, float], step: int) -> dict:
        sid = skill["skill_id"]
        without = {a.check: a.mean_score() for a in _select(arms, "retest_without", sid)}
        # where the skill is in the retrieved index, the control arm is its "with" arm
        with_arms = [a for a in _select(arms, "control") if a.check in without] + _select(arms, "retest_with", sid)
        result = self._credit(with_arms, (sid,), {**baseline, **without})
        return {"skill_id": sid, "name": skill["name"], "step": step, **result.as_dict()}

    def _merge(self, arms: list[_Arm], tests: _Tests, family: str, step: int) -> dict[str, Any]:
        without = {a.check: a.mean_score() for a in _select(arms, "merge_without")}
        members = self._credit(_select(arms, "merge_members"), tuple(tests.member_ids), without)
        merged = self._credit(_select(arms, "merge_merged"), ("merged",), without)
        draft = tests.merged
        return {
            "member_ids": tests.member_ids,
            "family": family,
            "step": step,
            "merged": {k: draft[k] for k in ("name", "when_to_use", "procedure")} if draft else None,
            "merged_credit": merged.value,
            "members_credit": members.value,
            "merged_detail": merged.as_dict(),
            "members_detail": members.as_dict(),
        }

    # ------------------------------------------------------------------ validation group

    async def _run_eval_group(self, prompt: dict, sampling_params: dict, n: int) -> tuple[list[_Row], dict]:
        """Validation: episodes on the source task with the frozen library (read-only, no writing)."""
        source = TaskSpec.from_dict(prompt["extra_info"]["source"])
        agent = self._new_agent_loop()
        index, version = ([], -1)
        if self.skill_cfg.val_use_skills:
            index, version = await self._retrieve(source)
        arm = _Arm("eval", "eval", source, _with_handles(index), n)
        await self._run_arms(agent, [arm], sampling_params)
        rows, episodes = [], []
        for i, result in enumerate(arm.results):
            info = {
                "success": float(result.won),
                "steps": result.steps,
                "index_size": len(index),
                "loaded_skill": float(bool(result.loads)),
                "invalid_actions": result.invalid_actions,
            }
            rows.append(_Row(i, arm.name, result.output, result.score, info, _episode_meta(arm, result)))
            episodes.append(_episode_record(i, result, usage={}, writes=[]))
        record = {
            "uid": prompt["uid"],
            "kind": "eval_group",
            "family": source.family,
            "library_version": version,
            "source": _task_record(source),
            "arms": [arm.record(episodes)],
        }
        return rows, record

    # ------------------------------------------------------------------ TransferQueue / dumps

    async def _put_rows(self, prompt: dict, rows: list[_Row], validate: bool) -> None:
        """Write a unit's episodes to TransferQueue, each with its own reward and GRPO group.

        Mirrors ``AgentLoopWorkerTQBase._agent_loop_postprocess`` except that rewards are per
        episode and that extra columns carry the credit assignment to the trainer: ``adv_group``
        (the arm), ``write_mask`` / ``write_adv`` (token level, aligned with ``response_mask``) and
        ``skill_meta`` (json, for metrics).
        """
        uid = prompt["uid"]
        sample_fields = {k: v for k, v in prompt.items() if k != "agent_name"}
        keys, fields, tags = [], [], []
        for row in rows:
            output = row.output
            output.reward_score = float(row.reward)
            output.extra_fields["reward_extra_info"] = row.info

            prompt_ids = torch.tensor(output.prompt_ids, dtype=torch.int64)
            response_ids = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompt_ids, response_ids], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            item = output.as_dict()
            item.update(sample_fields)
            item.pop("multi_modal_data", None)
            item["session_id"] = row.session_id
            item["loss_mask"] = item["response_mask"]
            item["input_ids"] = input_ids
            item["position_ids"] = position_ids
            item["multi_modal_inputs"] = multi_modal_inputs
            item["adv_group"] = row.adv_group
            item["skill_meta"] = json.dumps(row.meta, default=str)
            write_mask = torch.zeros_like(item["response_mask"], dtype=torch.int64)
            write_adv = torch.zeros_like(item["response_mask"], dtype=torch.float32)
            for start, end, advantage in row.write_spans:
                span = slice(start, min(end, write_mask.numel()))
                write_mask[span] = item["response_mask"][span]
                write_adv[span] = advantage * item["response_mask"][span]
            item["write_mask"], item["write_adv"] = write_mask, write_adv

            keys.append(f"{uid}_{row.session_id}_0")
            fields.append(item)
            prompt_len, response_len = item["prompts"].size(0), item["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    "global_steps": prompt["global_steps"],
                    "min_global_steps": item["extra_fields"].get("min_global_steps"),
                    "max_global_steps": item["extra_fields"].get("max_global_steps"),
                    "skill_role": row.meta.get("role", ""),
                }
            )
        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )

    def _dump(self, record: dict, trajectory: dict) -> None:
        """Append the human-readable unit record (messages, skills, credits, rewards) to a JSONL file."""
        dump_dir = self.skill_cfg.dump_dir
        if not dump_dir:
            return
        if self._dump_fh is None:
            os.makedirs(dump_dir, exist_ok=True)
            self._dump_fh = open(os.path.join(dump_dir, f"groups_worker{os.getpid()}.jsonl"), "a")
        record = {"step": int(trajectory["step"]), "validate": bool(trajectory["validate"]), **record}
        self._dump_fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._dump_fh.flush()


class SkillLoopManagerTQ(AgentLoopManagerTQ):
    """AgentLoopManager whose workers run skill-evolution groups; it also owns the library actor.

    Enable with::

        +actor_rollout_ref.rollout.agent.agent_loop_manager_class=\
verl.experimental.skill_loop.skill_loop_tq.SkillLoopManagerTQ
    """

    agent_loop_worker_cls = SkillLoopWorkerTQ

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = SkillLoopConfig.from_config(self.config)
        # Created before the workers; the manager keeps the handle so the actor outlives every rollout.
        self.skill_library = create_skill_library_actor(cfg.library, cfg.library_path)


async def _gather_settled(coros: list) -> list:
    """``asyncio.gather`` that lets every sibling finish before re-raising the first error.

    A group is reported as failed only after all of its episodes stopped touching the LLM server
    and the environment thread (same reason as upstream's ``_settle_session_tasks``).
    """
    results = await asyncio.gather(*coros, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _select(arms: list[_Arm], role: str, subject: str = "") -> list[_Arm]:
    return [a for a in arms if a.role == role and (not subject or a.subject == subject)]


def _with_handles(entries: list[dict[str, Any]], extra=(), seed: str = "") -> list[dict[str, Any]]:
    """Index as shown in a prompt: retrieved entries (ordered by utility) plus ``extra`` skills
    inserted at seeded random positions, so a skill under test does not own a fixed line."""
    index = [dict(e) for e in entries]
    rng = random.Random(seed)
    for e in extra:
        index.insert(rng.randint(0, len(index)), dict(e))
    for i, e in enumerate(index):
        e["handle"] = f"S{i + 1}"
    return index


def _task_record(task: TaskSpec) -> dict[str, Any]:
    return {"task_id": task.task_id, "family": task.family, "goal": task.payload.get("goal", "")}


def _episode_meta(arm: _Arm, result: SolveResult) -> dict[str, Any]:
    return {
        "role": arm.role,
        "arm": arm.name,
        "score": result.score,
        "won": float(result.won),
        "steps": result.steps,
        "n_loads": len(result.loads),
    }


def _episode_record(session_id: Optional[int], result: SolveResult, usage: dict, writes: list[dict]) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "won": result.won,
        "steps": result.steps,
        "invalid_actions": result.invalid_actions,
        "reward": result.score,
        "loads": result.loads,
        "usage": usage,
        "writes": writes,
        "messages": result.messages,
    }


def _load_report(arms: list[_Arm]) -> list[dict[str, Any]]:
    """Ordinary use of library skills: how often each was loaded and how those episodes did
    relative to their arm (feeds the re-test priority, never a reward)."""
    report: dict[str, dict[str, Any]] = {}
    for arm in arms:
        mean = arm.mean_score()
        for result in arm.results:
            for load in result.loads:
                entry = report.setdefault(load["skill_id"], {"skill_id": load["skill_id"], "n": 0, "outcome_sum": 0.0})
                entry["n"] += 1
                entry["outcome_sum"] += result.score - mean
    return list(report.values())
