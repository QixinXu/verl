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
"""Agent loop of skill-based self-evolution.

``SkillAgentLoop.solve`` runs one episode in a text environment. Next to the environment's own
commands the policy has two skill actions, both voluntary and available at any turn:

- ``load_skill S<k>``: the body of an indexed skill is returned as the next observation;
- ``write_skill`` + ``Name / When to use / Procedure``: the skill becomes a *candidate*. The
  episode goes on; whether the skill enters the library is decided by its credit on check tasks.

Both are ordinary actions of the trajectory: they cost a step and their tokens are trained. The
loop also records what the credit assignment needs: the token span of every ``write_skill`` call
(its reward differs from the rest of the trajectory) and, for the skills under test, the
*influence* of a loaded body on the tokens generated after it.

The class does not decide which episodes to run or how they are rewarded: that is a property of a
whole group and lives in ``SkillLoopWorkerTQ``. ``run`` is the plain single-episode entry point, so
the class also works under the stock ``AgentLoopWorker`` (no skills, task reward only).
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.skill_loop import credit, prompts
from verl.experimental.skill_loop.envs.base import TaskSpec, episode_score
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TextWorld keeps process-global parser state, so every environment call of a worker process goes
# through one thread. Parallelism comes from running several agent loop workers (processes).
_ENV_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="skill-loop-env")


@dataclass
class WrittenSkill:
    skill: dict[str, str]
    """``{"name", "when_to_use", "procedure"}``"""
    span: tuple[int, int]
    """Response-coordinate token span of the whole ``write_skill`` turn (thought, call, arguments)."""
    step: int
    """Step of the episode at which it was written."""


@dataclass
class SolveResult:
    output: AgentLoopOutput
    messages: list[dict[str, str]]
    won: bool
    steps: int
    score: float
    invalid_actions: int
    loads: list[dict[str, Any]] = field(default_factory=list)
    """Index entries the policy loaded, in order (``handle``, ``skill_id``, ``name``, ``step``)."""
    writes: list[WrittenSkill] = field(default_factory=list)
    influence: dict[str, Optional[float]] = field(default_factory=dict)
    """skill_id -> influence, for every loaded skill that was to be measured."""

    def loaded(self, skill_id: str) -> bool:
        return any(e["skill_id"] == skill_id for e in self.loads)


@register("skill_agent")
class SkillAgentLoop(AgentLoopBase):
    """Text-environment agent that can load skills and write skills."""

    merges_context_messages = True  # solve() appends user turns to the generated tokens

    def __init__(
        self,
        *args,
        env_module: str = "verl.experimental.skill_loop.envs.alfworld_env",
        env_kwargs: Optional[dict[str, Any]] = None,
        max_turn_tokens: int = 256,
        max_skill_words: int = 120,
        efficiency_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_module = importlib.import_module(env_module)
        self.env_kwargs = dict(env_kwargs or {})
        self.max_turn_tokens = max_turn_tokens
        self.max_skill_words = max_skill_words
        self.efficiency_weight = efficiency_weight
        self.response_length = self.rollout_config.response_length
        # ChatML-style models end an assistant turn with their EOS token (<|im_end|> for Qwen).
        self.end_of_turn_id = self.tokenizer.eos_token_id

    # ------------------------------------------------------------------ token bookkeeping

    async def _generate(self, state: _TokenState, sampling_params: dict[str, Any], max_tokens: int) -> str:
        """One assistant turn: token-in-token-out, then fold the tokens into the running trajectory."""
        budget = min(max_tokens, self.response_length - len(state.response_mask))
        params = {**sampling_params, "max_tokens": max(budget, 1)}
        with simple_timer("generate_sequences", state.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=state.request_id, prompt_ids=state.token_ids, sampling_params=params
            )
        preempted = output.num_preempted if output.num_preempted is not None else 0
        state.metrics["num_preempted"] = state.metrics.get("num_preempted", 0) + preempted
        if not state.extra_fields:
            state.extra_fields.update(output.extra_fields)
        elif output.extra_fields.get("max_global_steps"):
            # a trajectory that spans several weight versions keeps its first min and its last max
            state.extra_fields["max_global_steps"] = output.extra_fields["max_global_steps"]

        has_logprobs = bool(output.log_probs)
        merge_result, state.response_mask, logprobs = await self.ct_merge_assistant_token(
            state.token_ids,
            output.token_ids,
            state.response_mask,
            state.response_logprobs if (state.response_logprobs or has_logprobs) else None,
            assistant_logprobs=output.log_probs if has_logprobs else None,
        )
        state.token_ids = merge_result.token_ids
        if logprobs is not None:
            state.response_logprobs = logprobs
        # A turn cut off by the per-turn token cap (or by a server that strips the stop token) has
        # no end-of-turn token. Close it ourselves, as context (mask 0): otherwise the next user
        # turn would be appended to an unterminated assistant turn and the rest of the episode
        # would be generated, and trained, on a malformed conversation.
        if self.end_of_turn_id is not None and (not output.token_ids or output.token_ids[-1] != self.end_of_turn_id):
            state.token_ids = [*state.token_ids, self.end_of_turn_id]
            state.response_mask = [*state.response_mask, 0]
            if state.response_logprobs:
                state.response_logprobs = [*state.response_logprobs, 0.0]
        return self.tokenizer.decode(output.token_ids, skip_special_tokens=True).strip()

    async def _append_user(self, state: _TokenState, messages: list[dict], content: str) -> None:
        previous = list(messages)
        messages.append({"role": "user", "content": content})
        merge_result, state.response_mask, logprobs = await self.ct_merge_context_msg(
            previous, messages, state.token_ids, state.response_mask, state.response_logprobs or None
        )
        state.token_ids = merge_result.token_ids
        if state.response_logprobs:
            state.response_logprobs = logprobs or []

    def _finalize(self, state: _TokenState, num_turns: int) -> AgentLoopOutput:
        n_resp = len(state.response_mask)
        response_ids = state.token_ids[len(state.token_ids) - n_resp :] if n_resp else []
        prompt_ids = state.token_ids[: len(state.token_ids) - n_resp]
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0, "num_preempted": 0, **state.metrics}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=state.response_mask[: self.response_length],
            response_logprobs=state.response_logprobs[: self.response_length] if state.response_logprobs else None,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=dict(state.extra_fields),
        )
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return output

    async def _score(self, token_ids: list[int]) -> list[float]:
        """Teacher-forced log-prob of every token of ``token_ids`` given the tokens before it.

        One forward pass, nothing is sampled: the server is asked for the log-probs of the prompt
        itself. Position 0 has no context and gets 0.
        """
        output: TokenOutput = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=list(token_ids),
            sampling_params={"prompt_logprobs": 0, "max_tokens": 1},
        )
        rows = output.extra_fields["prompt_logprobs"]  # rows[t - 1] belongs to token t
        return [0.0] + [float(rows[t][0]) for t in range(len(token_ids) - 1)]

    async def complete(self, messages: list[dict], sampling_params: dict[str, Any], max_tokens: int) -> str:
        """A plain completion that is not part of any trained trajectory (used to draft merged skills)."""
        token_ids = await self.ct_build_initial_tokens(messages)
        output: TokenOutput = await self.server_manager.generate(
            request_id=uuid4().hex, prompt_ids=token_ids, sampling_params={**sampling_params, "max_tokens": max_tokens}
        )
        return self.tokenizer.decode(output.token_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ episode

    async def solve(
        self,
        task: TaskSpec,
        sampling_params: dict[str, Any],
        *,
        index: Optional[list[dict[str, Any]]] = None,
        measure: tuple[str, ...] = (),
    ) -> SolveResult:
        """Run one episode on ``task``.

        ``index``: entries ``{handle, skill_id, name, when_to_use, text}``. Name and when-to-use
        are in the prompt; ``load_skill <handle>`` returns the text.
        ``write_skill`` is offered on every task; when the task is completed the policy gets one more
        turn, so it can write knowing that it succeeded. Only the caller decides which writes count.
        ``measure``: skill ids whose influence is computed if the policy loads them.
        """
        loop = asyncio.get_running_loop()
        env = self.env_module.make_env(task, **self.env_kwargs)
        state = _TokenState(request_id=uuid4().hex)
        episode = _Episode()
        index = list(index or [])
        by_handle = {e["handle"].lower(): e for e in index}
        segments: dict[str, tuple[int, int, list[int]]] = {}
        completed = False  # the task is solved and the policy has its one extra turn to write a skill
        try:
            with simple_timer("tool_calls", state.metrics):
                first_obs = await loop.run_in_executor(_ENV_EXECUTOR, env.reset)
            messages = prompts.build_solve_messages(
                first_obs, self.env_module.ACTION_GUIDE, index, self.max_skill_words
            )
            state.token_ids = await self.ct_build_initial_tokens(messages)

            while episode.steps < env.max_steps:
                # stop early when the context budget cannot hold another full turn
                if len(state.response_mask) + self.max_turn_tokens >= self.response_length:
                    break
                span_start = len(state.response_mask)
                reply = await self._generate(state, sampling_params, self.max_turn_tokens)
                span = (span_start, len(state.response_mask))
                messages.append({"role": "assistant", "content": reply})
                command = prompts.parse_action(reply)
                skill_text = prompts.split_write(reply)
                loaded_entry, done = None, False

                # Skill actions are handled here; the environment never sees them.
                if skill_text is not None:
                    observation = self._write_skill(episode, skill_text, span)
                    done = completed
                elif completed:
                    break  # the extra turn was not used to write
                elif command is None:
                    # a malformed reply still costs a step: formatting is part of the task
                    episode.steps += 1
                    episode.invalid_actions += 1
                    observation = prompts.FORMAT_ERROR
                elif (handle := prompts.parse_load(command)) is not None:
                    observation, loaded_entry = self._load_skill(episode, handle, by_handle)
                else:
                    t0 = loop.time()
                    result = await loop.run_in_executor(_ENV_EXECUTOR, env.step, command)
                    state.metrics["tool_calls"] = state.metrics.get("tool_calls", 0.0) + loop.time() - t0
                    episode.steps += 1
                    episode.invalid_actions += int(not result.valid)
                    episode.won = result.won
                    observation, done = result.observation, result.done
                    if result.won and episode.steps < env.max_steps:
                        completed, done = True, False
                        observation = f"{observation}\n{prompts.TASK_COMPLETED}"
                if done or episode.steps >= env.max_steps:
                    break

                before = (list(messages), list(state.token_ids), list(state.response_mask))
                await self._append_user(state, messages, observation)
                if loaded_entry is not None and loaded_entry["skill_id"] in measure:
                    segments[loaded_entry["skill_id"]] = await self._blank_segment(
                        *before, state.token_ids, loaded_entry
                    )
        finally:
            await loop.run_in_executor(_ENV_EXECUTOR, env.close)

        return SolveResult(
            output=self._finalize(state, num_turns=len(messages)),
            messages=messages,
            won=episode.won,
            steps=episode.steps,
            score=episode_score(episode.won, episode.steps, env.max_steps, self.efficiency_weight),
            invalid_actions=episode.invalid_actions,
            loads=episode.loads,
            writes=episode.writes,
            influence=await self._influence(state, segments),
        )

    def _write_skill(self, episode: _Episode, skill_text: str, span: tuple[int, int]) -> str:
        """Handle a ``write_skill`` call; returns the observation. The call costs a step either way."""
        episode.steps += 1
        skill = prompts.parse_skill(skill_text, self.max_skill_words)
        if skill is None:
            episode.invalid_actions += 1
            return prompts.WRITE_MALFORMED
        episode.writes.append(WrittenSkill(skill=skill, span=span, step=episode.steps - 1))
        return prompts.WRITE_SAVED

    def _load_skill(
        self, episode: _Episode, handle: str, by_handle: dict[str, dict[str, Any]]
    ) -> tuple[str, Optional[dict[str, Any]]]:
        """Handle a ``load_skill`` call; returns the observation and the entry if its body was shown."""
        episode.steps += 1
        entry = by_handle.get(handle)
        if entry is None:
            episode.invalid_actions += 1
            return prompts.LOAD_UNKNOWN.format(handle=handle.upper() or "?"), None
        if any(e["skill_id"] == entry["skill_id"] for e in episode.loads):
            return prompts.LOAD_AGAIN.format(handle=entry["handle"]), None
        episode.loads.append({k: entry[k] for k in ("handle", "skill_id", "name")} | {"step": episode.steps - 1})
        return prompts.render_loaded(entry), entry

    # ------------------------------------------------------------------ influence of a loaded skill

    async def _blank_segment(
        self, messages: list[dict], ids: list[int], mask: list[int], real_ids: list[int], entry: dict[str, Any]
    ) -> tuple[int, int, list[int]]:
        """Tokens of the load observation with the body removed.

        Returns ``(start, end, blank_tokens)``: replacing ``real_ids[start:end]`` by
        ``blank_tokens`` gives the context "the skill was loaded but its body is empty".
        """
        blank = [*messages, {"role": "user", "content": prompts.render_loaded(entry, blank=True)}]
        merged, _, _ = await self.ct_merge_context_msg(messages, blank, ids, mask, None)
        blank_ids = list(merged.token_ids)
        start = 0
        while start < min(len(blank_ids), len(real_ids)) and blank_ids[start] == real_ids[start]:
            start += 1
        return start, len(real_ids), blank_ids[start:]

    async def _influence(
        self, state: _TokenState, segments: dict[str, tuple[int, int, list[int]]]
    ) -> dict[str, Optional[float]]:
        """For each measured skill: mean log-prob change of the tokens generated after its body.

        The actions are kept exactly as they were sampled; only the context changes (teacher
        forcing). It costs one forward pass for the real trajectory and one per measured skill.
        """
        if not segments:
            return {}
        token_ids = state.token_ids
        generated = [0] * (len(token_ids) - len(state.response_mask)) + list(state.response_mask)
        if not any(generated[end:] for _, end, _ in segments.values()):
            return dict.fromkeys(segments)  # nothing was generated after any of the loads
        with_body = await self._score(token_ids)
        out: dict[str, Optional[float]] = {}
        for skill_id, (start, end, blank_tokens) in segments.items():
            without_body = await self._score(token_ids[:start] + blank_tokens + token_ids[end:])
            out[skill_id] = credit.influence(with_body, without_body, generated, end, len(blank_tokens) - (end - start))
        return out

    # ------------------------------------------------------------------ stock entry point

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Single episode on the sample's source task, no skills. Reward = task score."""
        extra_info = kwargs.get("extra_info") or {}
        task = TaskSpec.from_dict(extra_info["source"])
        result = await self.solve(task, sampling_params)
        result.output.reward_score = result.score
        result.output.extra_fields["reward_extra_info"] = {"success": float(result.won), "steps": result.steps}
        return result.output


@dataclass
class _Episode:
    """Running outcome of one episode."""

    steps: int = 0
    invalid_actions: int = 0
    won: bool = False
    loads: list[dict[str, Any]] = field(default_factory=list)
    writes: list[WrittenSkill] = field(default_factory=list)


@dataclass
class _TokenState:
    """Running token-level state of one trajectory (the TITO contract: never re-encode decoded text)."""

    request_id: str
    token_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)
