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
"""End-to-end test of one skill-evolution unit: source task -> new skills -> check tasks -> credit -> library.

No model, no simulator, no Ray: the LLM is a scripted policy, the environment is ``toy_env`` and
TransferQueue is captured in memory. What is real is everything this package adds: the arms of a
unit, the credit of a skill, the token-level reward of ``write_skill`` calls and the library.
"""

import json
import random
import re

import pytest
from omegaconf import OmegaConf

import verl.experimental.skill_loop.skill_loop_tq as skill_loop_tq
from tests.experimental.skill_loop import toy_env
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.experimental.skill_loop.envs.base import TaskSpec
from verl.experimental.skill_loop.skill_agent_loop import SkillAgentLoop
from verl.experimental.skill_loop.skill_library import SkillLibrary
from verl.experimental.skill_loop.skill_loop_tq import SkillLoopConfig, SkillLoopWorkerTQ
from verl.workers.rollout.replica import TokenOutput


class _CharTokenizer:
    """One token per character: enough to exercise the token bookkeeping of the agent loop."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def _encode(text):
    return [ord(c) for c in text]


def _bodies(context):
    """Skill bodies that were loaded into a context (empty strings for blanked ones)."""
    return re.findall(r"Loaded skill S\d+:\n(.*?)\n\nContinue with the task\.", context, flags=re.DOTALL)


class _ScriptedLLM:
    """Policy with exactly the behaviour a skill is supposed to change.

    Loading: ``match`` loads the indexed skill whose when-to-use names the colour of its door,
    ``any`` loads the first indexed skill, ``never`` loads nothing.
    Acting: with ``follow`` it presses the button a loaded body tells it to press; otherwise, and
    without a body, it guesses, never repeating a button.
    Writing: with ``write``, once the task is completed it writes down the button that worked.

    Its "log-probs" encode the same behaviour: a press dictated by a loaded body is certain
    (-0.1 per token), a guessed press is one of three (-1.1 per token).
    """

    def __init__(self, seed=0, write=True, load="match", follow=True):
        self.rng = random.Random(seed)
        self.write, self.load, self.follow = write, load, follow

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        prompt = "".join(chr(i) for i in prompt_ids)
        if "prompt_logprobs" in sampling_params:
            return TokenOutput(token_ids=[0], log_probs=[0.0], extra_fields={"prompt_logprobs": self._score(prompt)})
        text = self._merge(prompt) if "Write the merged skill now." in prompt else self._act(prompt)
        ids = _encode(text)[: sampling_params["max_tokens"]]
        return TokenOutput(
            token_ids=ids, log_probs=[-0.1] * len(ids), extra_fields={"min_global_steps": 1, "max_global_steps": 1}
        )

    def _act(self, prompt):
        last_turn = prompt.split("<user>")[-1]
        colour = re.search(r"You face a (\w+) door", prompt).group(1)
        pressed = re.findall(r"Action: press ([abc])", prompt)
        if "Task completed" in last_turn:
            if not self.write:
                return "Thought: nothing to save\nAction: done"
            return (
                f"Thought: the button that worked is worth saving\nAction: write_skill\n"
                f"Name: Open {colour} doors\nWhen to use: the task is to open a {colour} door\n"
                f"Procedure:\n1. press {pressed[-1]}\n2. open door"
            )
        if "You press" in last_turn:
            return "Thought: try the door\nAction: open door"
        entries = re.findall(r"^\[(S\d+)\] (.+)$", prompt.split("<user>")[0], flags=re.MULTILINE)
        if entries and "Action: load_skill" not in prompt:
            if self.load == "match":
                pick = next((h for h, line in entries if f"{colour} door" in line), None)
            else:
                pick = entries[0][0] if self.load == "any" else None
            if pick:
                return f"Thought: this skill fits\nAction: load_skill {pick}"
        told = [b for body in _bodies(prompt) for b in re.findall(r"press ([abc])", body)]
        if self.follow and told and told[0] not in pressed:
            return f"Thought: the skill says so\nAction: press {told[0]}"
        button = self.rng.choice(sorted(set("abc") - set(pressed)) or ["a"])
        return f"Thought: guess\nAction: press {button}"

    def _merge(self, prompt):
        button = re.search(r"press ([abc])", prompt).group(1)
        colour = re.search(r"open a (\w+) door", prompt).group(1)
        return (
            f"Name: Open {colour} doors (merged)\nWhen to use: the task is to open a {colour} door\n"
            f"Procedure:\n1. press {button}\n2. open door"
        )

    def _score(self, prompt):
        logprobs = [-0.1] * len(prompt)
        for turn in re.finditer(r"<assistant>(.*?)(?=\n<user>|$)", prompt, flags=re.DOTALL):
            press = re.search(r"Action: press ([abc])", turn.group(1))
            if not press:
                continue
            context = prompt[: turn.start()]
            told = [b for body in _bodies(context) for b in re.findall(r"press ([abc])", body)]
            if not (self.follow and told and told[0] == press.group(1)):
                logprobs[turn.start(1) : turn.end(1)] = [-1.1] * (turn.end(1) - turn.start(1))
        return [[lp] for lp in logprobs[1:]] + [[0.0]]


class _ToyAgentLoop(SkillAgentLoop):
    """SkillAgentLoop with the tokenizer-dependent pieces replaced by a character-level renderer."""

    def __init__(self, llm, max_steps=6):
        self.env_module = toy_env
        self.env_kwargs = {"max_steps": max_steps}
        self.max_turn_tokens, self.max_skill_words = 200, 40
        self.efficiency_weight = 0.5
        self.response_length = 6000
        self.server_manager = llm
        self.tokenizer = _CharTokenizer()
        self.end_of_turn_id = None

    async def ct_build_initial_tokens(self, messages, **kwargs):
        return _encode("".join(f"<{m['role']}>{m['content']}\n" for m in messages) + "<assistant>")

    async def ct_merge_assistant_token(self, ids, new_ids, mask, logprobs=None, assistant_logprobs=None):
        result = type("R", (), {"token_ids": list(ids) + list(new_ids)})()
        return result, mask + [1] * len(new_ids), (logprobs or []) + list(assistant_logprobs or [])

    async def ct_merge_context_msg(self, previous, updated, ids, mask, logprobs=None, tools=None):
        added = _encode(f"\n<user>{updated[-1]['content']}\n<assistant>")
        result = type("R", (), {"token_ids": list(ids) + added})()
        return result, mask + [0] * len(added), (logprobs or []) + [0.0] * len(added)


class _Remote:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        async def call():
            return self.fn(*args, **kwargs)

        return call()


class _LibraryHandle:
    """Stands in for the Ray actor handle: ``await handle.method.remote(...)``."""

    def __init__(self, library: SkillLibrary):
        self._library = library

    def __getattr__(self, name):
        if name == "retrieve":
            lib = self._library
            return _Remote(lambda **kw: {"version": lib.version, "skills": lib.retrieve(**kw)})
        return _Remote(getattr(self._library, name))


def _skill(colour, button, extra=""):
    return {
        "name": f"Open {colour} doors",
        "when_to_use": f"the task is to open a {colour} door",
        "procedure": f"1. press {button}\n2. open door{extra}",
    }


class _Harness:
    def __init__(self, monkeypatch, n=6, llm=None, library=None, **skill_cfg):
        self.library = SkillLibrary({"embedder": "hash", **(library or {})})
        self.puts, self.status, self.records = [], [], []
        worker = object.__new__(SkillLoopWorkerTQ)
        worker.skill_cfg = SkillLoopConfig(**skill_cfg)
        worker.config = OmegaConf.create(
            {"actor_rollout_ref": {"rollout": {"n": n, "val_kwargs": {"n": 1}}}, "algorithm": {}}
        )
        worker._library = _LibraryHandle(self.library)
        worker._dump = lambda record, trajectory: self.records.append(record)
        worker.processor = None
        worker.mm_processor_kwargs = {}
        llm = llm or _ScriptedLLM()
        worker._new_agent_loop = lambda: _ToyAgentLoop(llm)
        worker._compute_multi_modal_inputs = lambda output, input_ids: AgentLoopWorker._compute_multi_modal_inputs(
            worker, output, input_ids
        )
        self.worker = worker

        async def kv_put(*, key, partition_id, tag):
            self.status.append((key, partition_id, tag["status"]))

        async def kv_batch_put(*, keys, fields, tags, partition_id):
            self.puts.append({"keys": keys, "fields": fields, "tags": tags, "partition_id": partition_id})

        monkeypatch.setattr(skill_loop_tq.tq, "async_kv_put", kv_put)
        monkeypatch.setattr(skill_loop_tq.tq, "async_kv_batch_put", kv_batch_put)

    def seed_skill(self, colour, button, extra="", step=0):
        skill = self.library._insert(_skill(colour, button, extra), colour, f"seed-{colour}-{button}{extra}", step)
        skill.utility = 0.3
        return skill

    @staticmethod
    def prompt(uid, family, task_id, check_family=None, with_checks=True, step=1):
        def task(tid, fam):
            return {"task_id": tid, "family": fam, "payload": {"goal": f"open the {fam} door"}}

        extra_info = {"index": 0, "source": task(task_id, family)}
        if with_checks:
            extra_info["probes"] = [task(f"{task_id}-check{j}", check_family or family) for j in range(2)]
        return {
            "uid": uid,
            "agent_name": "skill_agent",
            "global_steps": step,
            "raw_prompt": [{"role": "user", "content": "x"}],
            "extra_info": extra_info,
        }

    async def run(self, prompt, validate=False):
        params = {"temperature": 1.0, "logprobs": True}
        await self.worker._run_prompt(prompt, params, {"validate": validate, "step": prompt["global_steps"]})
        return self.puts[-1] if self.puts else None


def _rows(put):
    fields = put["fields"]
    rows = []
    for i, key in enumerate(put["keys"]):
        meta = json.loads(fields["skill_meta"][i])
        rows.append(
            {
                "key": key,
                "group": fields["adv_group"][i],
                "reward": fields["rm_scores"][i].sum().item(),
                "write_tokens": int(fields["write_mask"][i].sum().item()),
                "write_adv": fields["write_adv"][i].abs().max().item() * fields["write_adv"][i].sum().sign().item(),
                **meta,
            }
        )
    return rows


def _arm(record, name):
    return next(a for a in record["arms"] if a["name"] == name)


@pytest.mark.asyncio
async def test_a_unit_writes_a_skill_checks_it_and_rewards_the_write_call_with_its_credit(monkeypatch):
    h = _Harness(monkeypatch, n=6)
    put = await h.run(h.prompt("u1", "red", "red-1"))
    assert h.status == [("u1", "train", "running"), ("u1", "train", "finished")]
    record, rows = h.records[-1], _rows(put)

    # arms: 6 source episodes, and per check task 4 control + 4 episodes for the one new skill
    assert [a["name"] for a in record["arms"]] == [
        "source",
        "check0/control",
        "check0/new0",
        "check1/control",
        "check1/new0",
    ]
    assert len(rows) == 6 + 2 * (4 + 4) and len({r["key"] for r in rows}) == len(rows)

    # every source episode that succeeded early enough wrote the same skill: ONE candidate
    source = [r for r in rows if r["role"] == "source"]
    writers = [r for r in source if r["n_writes"]]
    assert writers and len(record["candidates"]) == 1
    cand = record["candidates"][0]
    assert cand["procedure"].startswith("1. press a") and cand["duplicates"] == len(writers) - 1

    # control episodes have no skill to load and guess; the episodes with the new skill load it and use it
    for j in range(2):
        assert all(not e["loads"] for e in _arm(record, f"check{j}/control")["episodes"])
        for e in _arm(record, f"check{j}/new0")["episodes"]:
            usage = e["usage"]["new0"]
            assert usage["loaded"] and usage["influence"] > 0.1 and usage["weight"] == 1.0
    # credit = sum u * (r - baseline of that check task) / (sum u + kappa), here with u = 1 for all 8
    expected = sum(
        e["reward"] - record["baseline"][str(j)] for j in range(2) for e in _arm(record, f"check{j}/new0")["episodes"]
    ) / (8 + 1.0)
    assert cand["credit"] == pytest.approx(expected, abs=1e-5) and cand["credit"] > 0

    # reward: every episode keeps its own task score ...
    for r in rows:
        assert r["reward"] == pytest.approx(r["score"])
    # ... and only the write_skill tokens of the writers carry the (normalized) credit
    for r in source:
        assert (r["write_tokens"] > 0) == bool(r["n_writes"])
        assert (r["write_adv"] > 0) if r["n_writes"] else (r["write_adv"] == 0)
    assert all(r["write_tokens"] == 0 for r in rows if r["role"] != "source")

    summary = h.library.commit(step=1)
    assert summary["admitted"] == 1 and summary["num_active"] == 1
    skill = next(iter(h.library.skills.values()))
    assert skill.family == "red" and skill.utility == pytest.approx(cand["credit"], abs=1e-5)


@pytest.mark.asyncio
async def test_a_skill_that_is_not_loaded_on_the_check_tasks_earns_nothing(monkeypatch):
    h = _Harness(monkeypatch, n=6)
    # the skill is written on a red door but the check tasks are blue doors: its when-to-use does
    # not match, the policy does not load it
    put = await h.run(h.prompt("u1", "red", "red-1", check_family="blue"))
    record = h.records[-1]
    cand = record["candidates"][0]
    assert cand["n_loaded"] == 0 and cand["credit"] == 0.0
    assert all(r["write_adv"] == 0 for r in _rows(put))
    assert h.library.commit(step=1)["num_active"] == 0


@pytest.mark.asyncio
async def test_a_skill_that_misleads_on_the_check_tasks_gets_negative_credit(monkeypatch):
    h = _Harness(monkeypatch, n=6, llm=_ScriptedLLM(load="any"))
    # this policy loads whatever is indexed: on blue doors the red-door skill sends it to the wrong button
    put = await h.run(h.prompt("u1", "red", "red-1", check_family="blue"))
    cand = h.records[-1]["candidates"][0]
    assert cand["n_loaded"] == 8 and cand["credit"] < 0
    writers = [r for r in _rows(put) if r["role"] == "source" and r["n_writes"]]
    assert writers and all(r["write_adv"] < 0 for r in writers)
    assert h.library.commit(step=1)["num_active"] == 0


@pytest.mark.asyncio
async def test_a_skill_that_is_loaded_but_ignored_gets_no_credit(monkeypatch):
    h = _Harness(monkeypatch, n=6, llm=_ScriptedLLM(follow=False))
    await h.run(h.prompt("u1", "red", "red-1"))
    record = h.records[-1]
    cand = record["candidates"][0]
    # loaded in all 8 episodes, yet the actions are as likely without the body as with it
    assert cand["n_loaded"] == 8 and cand["weight"] == 0.0 and cand["credit"] == 0.0
    usage = [e["usage"]["new0"] for j in range(2) for e in _arm(record, f"check{j}/new0")["episodes"]]
    assert all(u["loaded"] and u["influence"] == pytest.approx(0.0) and u["weight"] == 0.0 for u in usage)


@pytest.mark.asyncio
async def test_tau_zero_counts_every_load_as_full_use(monkeypatch):
    h = _Harness(monkeypatch, n=6, llm=_ScriptedLLM(follow=False), tau=0.0)
    await h.run(h.prompt("u1", "red", "red-1"))
    assert h.records[-1]["candidates"][0]["weight"] == 8.0


@pytest.mark.asyncio
async def test_a_second_unit_retrieves_the_library_skill_and_a_duplicate_adds_nothing(monkeypatch):
    h = _Harness(monkeypatch, n=6)
    await h.run(h.prompt("u1", "red", "red-1"))
    h.library.commit(step=1)
    first = next(iter(h.library.skills.values()))

    await h.run(h.prompt("u2", "red", "red-2", step=2))
    record = h.records[-1]
    source = _arm(record, "source")
    # the library skill is in the index of the source task and of the check tasks, and it is loaded
    assert [e["skill_id"] for e in source["index"]] == [first.skill_id]
    assert all(e["loads"] and e["loads"][0]["skill_id"] == first.skill_id for e in source["episodes"])
    assert [e["skill_id"] for e in _arm(record, "check0/control")["index"]] == [first.skill_id]
    # the same skill written again is measured AGAINST the library that already has it: about zero
    cand = record["candidates"][0]
    assert abs(cand["credit"]) < 0.1
    summary = h.library.commit(step=2)
    assert summary["num_active"] == 1 and summary.get("admitted", 0) == 0
    # ordinary use is counted: 6 source episodes and 8 control episodes loaded the skill
    assert first.n_loads == 6 + 8


@pytest.mark.asyncio
async def test_a_library_skill_is_retested_with_the_same_estimator_and_deleted_after_two_bad_retests(monkeypatch):
    h = _Harness(monkeypatch, n=4, llm=_ScriptedLLM(write=False))
    good, bad = h.seed_skill("red", "a"), h.seed_skill("blue", "c")  # blue doors open with b, not c

    # step 2: both are due (verified at step 0, retest_every = 2)
    await h.run(h.prompt("u1", "red", "red-1", step=2))
    record = h.records[-1]
    # the skill is in the retrieved index, so the control arm is the "with" arm and one extra
    # arm runs the same index without it
    without = _arm(record, f"check0/retest_{good.skill_id}/without")
    assert good.skill_id not in [e["skill_id"] for e in without["index"]] and len(without["episodes"]) == 4
    assert record["retests"][0]["skill_id"] == good.skill_id and record["retests"][0]["credit"] > 0
    await h.run(h.prompt("u2", "blue", "blue-1", step=2))
    assert h.records[-1]["retests"][0]["credit"] < 0
    summary = h.library.commit(step=2)
    assert summary["retested"] == 2 and good.utility > 0 and good.verified_step == 2 and good.strikes == 0
    assert bad.status == "active" and bad.strikes == 1 and bad.utility < 0

    # step 4: the bad skill fails its second re-test in a row and is deleted
    await h.run(h.prompt("u3", "blue", "blue-2", step=4))
    summary = h.library.commit(step=4)
    assert bad.status == "deleted" and bad.removed_reason == "retest_not_positive"
    assert good.status == "active"


@pytest.mark.asyncio
async def test_a_retested_skill_that_is_not_retrieved_is_put_into_the_index(monkeypatch):
    h = _Harness(monkeypatch, n=4, llm=_ScriptedLLM(write=False), library={"index_size": 0})
    good = h.seed_skill("red", "a")
    await h.run(h.prompt("u1", "red", "red-1", step=2))
    record = h.records[-1]
    with_arm = _arm(record, f"check0/retest_{good.skill_id}/with")
    assert [e["skill_id"] for e in with_arm["index"]] == [good.skill_id]
    assert _arm(record, "check0/control")["index"] == [] and record["retests"][0]["credit"] > 0


@pytest.mark.asyncio
async def test_a_merge_is_drafted_and_judged_against_its_cluster(monkeypatch):
    h = _Harness(
        monkeypatch,
        n=4,
        llm=_ScriptedLLM(write=False),
        library={"merge_every": 4, "merge_threshold": 0.5, "retest_every": 0},
    )
    a, b = h.seed_skill("red", "a"), h.seed_skill("red", "a", extra="\n3. walk through")
    await h.run(h.prompt("u1", "red", "red-1", step=4))
    record = h.records[-1]
    merge = record["merge"]
    assert merge["member_ids"] == [a.skill_id, b.skill_id] and merge["merged"]["name"].endswith("(merged)")
    # three arms per check task: the index without the cluster, with its members, with the merged skill
    assert [e["skill_id"] for e in _arm(record, "check0/merge/without")["index"]] == []
    assert sorted(e["skill_id"] for e in _arm(record, "check0/merge/members")["index"]) == [a.skill_id, b.skill_id]
    assert [e["skill_id"] for e in _arm(record, "check0/merge/merged")["index"]] == ["merged"]
    # merge arms only measure: their episodes are recorded but never become training rows
    assert all(e["session_id"] is None for e in _arm(record, "check0/merge/merged")["episodes"])
    assert merge["merged_credit"] > 0 and merge["members_credit"] > 0

    summary = h.library.commit(step=4)
    if merge["merged_credit"] >= merge["members_credit"]:
        assert summary["merged"] == 1 and a.status == b.status == "merged" and summary["num_active"] == 1
    else:
        assert summary["merge_rejected"] == 1 and a.status == b.status == "active"


@pytest.mark.asyncio
async def test_validation_group_is_read_only(monkeypatch):
    h = _Harness(monkeypatch, n=6)
    await h.run(h.prompt("u1", "red", "red-1"))
    h.library.commit(step=1)
    put = await h.run(h.prompt("v1", "red", "red-9", with_checks=False), validate=True)
    rows = _rows(put)
    assert put["partition_id"] == "val" and [r["role"] for r in rows] == ["eval"]
    # the frozen library is retrieved; the policy loads the red-door skill (1 step), wins in 3 and
    # writes a skill after completing (1 more step); that skill is dropped: validation stages nothing
    assert (
        rows[0]["n_loads"] == 1 and rows[0]["reward"] == pytest.approx(1 - 0.5 * 4 / 6) and rows[0]["write_tokens"] == 0
    )
    info = put["fields"]["extra_fields"][0]["reward_extra_info"]
    assert info["success"] == 1.0 and info["index_size"] == 1 and info["loaded_skill"] == 1.0
    assert not h.library._staged


@pytest.mark.asyncio
async def test_failed_group_reports_failure_and_writes_nothing(monkeypatch):
    h = _Harness(monkeypatch, n=4)
    bad = h.prompt("u1", "purple", "purple-1")  # the toy environment has no purple door
    assert await h.run(bad) is None
    assert h.status[-1] == ("u1", "train", "failure") and h.puts == []


class _FixedLLM:
    """Replies with a fixed list of turns; scores every token at -0.1."""

    def __init__(self, replies):
        self.replies = list(replies)

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        if "prompt_logprobs" in sampling_params:
            return TokenOutput(token_ids=[0], extra_fields={"prompt_logprobs": [[-0.1]] * len(prompt_ids)})
        ids = _encode(self.replies.pop(0))
        return TokenOutput(token_ids=ids, log_probs=[-0.1] * len(ids), extra_fields={})


_INDEX = [
    {
        "handle": "S1",
        "skill_id": "sk1",
        "name": "Open red doors",
        "when_to_use": "a red door",
        "text": "### Open red doors\n1. press a",
    }
]
_WRITE = "Thought: save it\nAction: write_skill\nName: Red\nWhen to use: a red door\nProcedure:\n1. press a"


async def _solve(replies, **kwargs):
    agent = _ToyAgentLoop(_FixedLLM([f"Thought: x\nAction: {r}" if "Action:" not in r else r for r in replies]))
    task = TaskSpec(task_id="red-1", family="red", payload={})
    return await agent.solve(task, {"temperature": 1.0}, **kwargs)


@pytest.mark.asyncio
async def test_skill_actions_are_ordinary_steps_of_the_episode():
    # loading takes a step and shows the body as the next observation
    r = await _solve(["load_skill S1", "press a", "open door", "done"], index=_INDEX)
    assert r.won and r.steps == 3 and r.invalid_actions == 0 and r.loaded("sk1")
    assert r.messages[3]["content"].startswith("Loaded skill S1:\n### Open red doors")
    # a skill can be loaded at any turn, also after a failed attempt
    r = await _solve(["press b", "open door", "load_skill(s1)", "press a", "open door", "done"], index=_INDEX)
    assert r.won and r.steps == 5 and r.loads[0]["step"] == 2
    # unknown handles are invalid actions; loading the same skill twice is not shown twice
    r = await _solve(["load_skill S7", "load_skill S1", "load_skill S1", "press a", "open door", "done"], index=_INDEX)
    assert r.won and r.steps == 5 and r.invalid_actions == 1 and len(r.loads) == 1
    assert "already loaded" in r.messages[7]["content"]


@pytest.mark.asyncio
async def test_write_skill_is_voluntary_can_happen_mid_episode_and_after_completion():
    # mid-episode: the skill is recorded with the token span of the whole turn, the episode goes on
    r = await _solve(["press a", _WRITE, "open door", "done"])
    assert r.won and r.steps == 3 and [w.step for w in r.writes] == [1]
    start, end = r.writes[0].span
    response = r.output.response_ids[start:end]
    assert "".join(chr(i) for i in response) == _WRITE and all(r.output.response_mask[start:end])
    # after completion the policy gets one more turn; `done` (or anything else) just ends the episode
    assert "Task completed" in r.messages[-2]["content"] and r.messages[-1]["content"].endswith("done")

    # after completion: writing costs a step like everywhere else
    r = await _solve(["press a", "open door", _WRITE])
    assert r.won and r.steps == 3 and len(r.writes) == 1 and r.score == pytest.approx(1 - 0.5 * 3 / 6)
    r = await _solve(["press a", "open door", "done"])
    assert r.won and r.steps == 2 and not r.writes

    # a malformed skill is not a candidate; there is no cap on the number of writes
    r = await _solve(["Thought: x\nAction: write_skill\njust press a", "press a", "open door", "done"])
    assert r.invalid_actions == 1 and not r.writes and "Skill not saved" in r.messages[3]["content"]
    r = await _solve([_WRITE, _WRITE, _WRITE, "press a", "open door", "done"])
    assert len(r.writes) == 3 and r.invalid_actions == 0 and r.won and r.steps == 5


@pytest.mark.asyncio
async def test_influence_compares_the_same_actions_with_and_without_the_loaded_body():
    llm = _ScriptedLLM(write=False)
    agent = _ToyAgentLoop(llm)
    task = TaskSpec(task_id="red-1", family="red", payload={})
    index = [{**_INDEX[0], "when_to_use": "the task is to open a red door"}]
    r = await agent.solve(task, {"temperature": 1.0}, index=index, measure=("sk1",))
    assert r.won and r.steps == 3 and r.loaded("sk1")
    # generated after the load: the press turn (certain with the body, a 1-in-3 guess without), the
    # open turn and the `done` of the extra turn after completion (the same either way)
    press, open_door = "Thought: the skill says so\nAction: press a", "Thought: try the door\nAction: open door"
    done = "Thought: nothing to save\nAction: done"
    assert r.influence["sk1"] == pytest.approx(len(press) * 1.0 / (len(press) + len(open_door) + len(done)))
    # not measured unless asked for, and nothing to measure without a load
    assert (await agent.solve(task, {"temperature": 1.0}, index=index)).influence == {}
    r = await _ToyAgentLoop(_ScriptedLLM(write=False, load="never")).solve(
        task, {"temperature": 1.0}, index=index, measure=("sk1",)
    )
    assert r.influence == {}


@pytest.mark.asyncio
async def test_a_turn_without_end_of_turn_token_is_closed_as_context():
    """A reply cut off by the per-turn cap must not leave the assistant turn open."""
    eot = 7

    class _LLM:
        def __init__(self, ends_with_eot):
            self.ends_with_eot = ends_with_eot

        async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
            ids = _encode("Thought: x\nAction: press a") + ([eot] if self.ends_with_eot else [])
            return TokenOutput(token_ids=ids, log_probs=[-0.1] * len(ids), extra_fields={})

    from verl.experimental.skill_loop.skill_agent_loop import _TokenState

    for ends_with_eot in (False, True):
        agent = _ToyAgentLoop(_LLM(ends_with_eot))
        agent.end_of_turn_id = eot
        state = _TokenState(request_id="r", token_ids=_encode("<user>hi\n<assistant>"))
        text = await agent._generate(state, {"temperature": 1.0}, max_tokens=64)
        assert text.startswith("Thought: x")
        assert state.token_ids[-1] == eot and state.token_ids.count(eot) == 1
        assert len(state.response_mask) == len(state.response_logprobs)
        # generated end-of-turn tokens are trainable, the one the loop had to add is context
        assert state.response_mask[-1] == (1 if ends_with_eot else 0)
        assert sum(state.response_mask) == len(_encode("Thought: x\nAction: press a")) + int(ends_with_eot)
