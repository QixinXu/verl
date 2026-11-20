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
"""A scripted stand-in for the LLM server, for driving the skill loop on a CPU node.

THIS IS NOT A MODEL. It is a hand-written ALFWorld policy with the same interface as verl's
``LLMServerClient`` (token ids in, token ids out), used to exercise the harness end to end without
a GPU: real tokenizer and chat template, real ALFWorld games, real Ray workers, TransferQueue,
replay buffer, advantages and library. Trajectories produced with it say nothing about what a
language model would do; they only show that the plumbing and the credit assignment are right.

To make the skill loop observable, the policy is deliberately *ignorant in ways a skill can fix*:

- it tries ``put``/``place`` (rejected by the game) before ``move``, unless a loaded skill mentions
  ``move <object> to``;
- it tries to heat on the stove before the microwave, unless a loaded skill mentions the microwave;
- it fumbles with the lamp before ``use``, unless a loaded skill mentions ``use <lamp>``;
- it wastes steps on random detours, less so when it has loaded any skill.

Skill actions, all voluntary:

- loading: at its first turn it is a noisy chooser (mostly an indexed skill whose when-to-use names
  its kind of task, sometimes a random one, sometimes none); later, after a rejected command, it
  may load one after all. Some episodes load a skill and then ignore it;
- writing: after the appliance step of a heat/clean/cool task worked it sometimes writes a skill
  mid-episode; after the task is completed it writes, at random, a correct procedure for the task
  type (in one of two wordings), a vague platitude, a malformed call, or nothing.

It also answers the log-prob requests the influence measurement makes. Its "log-probs" encode its
own behaviour: an action is likely (-0.2 per token) when it is what the policy would do in that
context and unlikely (-2.2) otherwise, so an action that only makes sense because of a loaded
body becomes unlikely once the body is blanked.
"""

from __future__ import annotations

import random
import re
from typing import Optional

from verl.workers.rollout.replica import TokenOutput

APPLIANCE = {"heat": ("microwave 1", "heat"), "cool": ("fridge 1", "cool"), "clean": ("sinkbasin 1", "clean")}

SEARCH_PRIORITY = {
    **dict.fromkeys(["countertop", "diningtable", "sidetable", "desk", "dresser", "coffeetable", "shelf", "bed"], 0),
    **dict.fromkeys(["sinkbasin", "sofa", "armchair", "toilet", "bathtubbasin", "stoveburner", "garbagecan"], 1),
    **dict.fromkeys(["fridge", "microwave"], 2),
    "drawer": 6,
    "cabinet": 7,
}

GOOD_SKILLS = {
    "pick": (
        "Find, take, place",
        "The task is to put an object in or on a receptacle",
        "1. Go to receptacles one at a time; open a closed one before looking inside.\n"
        "2. When you see the object, take <object> from <receptacle>.\n"
        "3. Go to the target receptacle.\n"
        "4. Place it with: move <object> to <receptacle> (the game rejects 'put').",
    ),
    "heat": (
        "Heat with the microwave, then place",
        "The task is to heat an object and put it somewhere",
        "1. Search receptacles until you see the object and take it.\n"
        "2. Go to the microwave (the stove burners cannot heat).\n"
        "3. Use: heat <object> with microwave.\n"
        "4. Go to the target receptacle and use: move <object> to <receptacle>.",
    ),
    "clean": (
        "Clean at the sink basin, then place",
        "The task is to clean an object and put it somewhere",
        "1. Search receptacles until you see the object and take it.\n"
        "2. Go to the sinkbasin and use: clean <object> with sinkbasin.\n"
        "3. Go to the target receptacle and use: move <object> to <receptacle>.",
    ),
    "look": (
        "Examine under a lamp",
        "The task is to look at an object under a lamp",
        "1. Search receptacles until you see the object and take it.\n"
        "2. Keep holding it and find a desklamp or floorlamp.\n"
        "3. Turn the lamp on with: use <lamp>.",
    ),
}
# a second wording of every good skill: similar enough to be merged, different enough not to be a duplicate
VARIANT_STEP = "If the game rejects a command, read the commands it lists as available and use one of those."
# words in a when-to-use line that tell the scripted chooser an entry is about its kind of task
FAMILY_CUES = {
    "pick": ["put an object"],
    "pick2": ["put an object"],
    "heat": ["heat"],
    "cool": ["cool", "heat"],
    "clean": ["clean"],
    "look": ["lamp"],
}
LOAD_MATCHING, LOAD_RANDOM = 0.6, 0.2  # at the first turn; the rest loads nothing
LOAD_WHEN_STUCK = 0.5  # after a rejected command, if nothing was loaded yet
IGNORE_LOADED = 0.15  # episodes that load a skill and then act as if they had not
WRITE_MID_EPISODE = 0.3  # after the appliance step worked
WRITE_GOOD, WRITE_VAGUE, WRITE_MALFORMED = 0.5, 0.2, 0.05  # after completion; the rest writes nothing
LIKELY, UNLIKELY, NEUTRAL = -0.2, -2.2, -0.5

VAGUE_SKILL = (
    "Be careful",
    "Any household task",
    "1. Read the task carefully.\n2. Explore the room.\n3. Do not give up.",
)


def _write_call(name: str, when: str, procedure: str, thought: str = "this is worth saving") -> str:
    return f"Thought: {thought}\nAction: write_skill\nName: {name}\nWhen to use: {when}\nProcedure:\n{procedure}"


def _good_skill(family: str, rng: random.Random) -> tuple[str, str, str]:
    name, when, procedure = GOOD_SKILLS[family]
    if rng.random() < 0.4:
        procedure = f"{procedure}\n{procedure.count(chr(10)) + 2}. {VARIANT_STEP}"
    return name, when, procedure


def _bodies(users: list[str]) -> str:
    """Bodies of the skills loaded so far (a blanked body contributes nothing)."""
    found = [re.search(r"^Loaded skill S\d+:\n(.*?)\n\nContinue with the task\.", u, flags=re.DOTALL) for u in users]
    return "\n".join(m.group(1) for m in found if m)


def parse_goal(goal: str) -> tuple[str, str, str]:
    """Return (family, object type, target receptacle or lamp type)."""
    goal = goal.strip().rstrip(".").lower()
    m = re.search(r"(heat|cool|clean) some (\w+) and put it in (\w+)", goal)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.search(r"(?:look at|examine the) (\w+) (?:under|with) the (\w+)", goal)
    if m:
        return "look", m.group(1), m.group(2)
    m = re.search(r"two (\w+) (?:in|on|and put them in) (\w+)", goal)
    if m:
        return "pick2", m.group(1), m.group(2)
    m = re.search(r"put (?:a|an|some) (\w+) (?:in|on) (\w+)", goal)
    if m:
        return "pick", m.group(1), m.group(2)
    return "pick", "", ""


class ScriptedAlfworldLLM:
    """Token-in/token-out scripted policy; pass it wherever verl expects an ``LLMServerClient``."""

    def __init__(self, model_path: str, seed: int = 0):
        self.model_path = model_path
        self.seed = seed
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self._tokenizer

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs) -> TokenOutput:
        if "prompt_logprobs" in sampling_params:
            return TokenOutput(
                token_ids=[self.tokenizer.convert_tokens_to_ids("<|im_end|>")],
                log_probs=[0.0],
                stop_reason="completed",
                num_preempted=0,
                extra_fields={"prompt_logprobs": self._score(list(prompt_ids))},
            )
        rng = random.Random(f"{self.seed}-{request_id}-{len(prompt_ids)}")
        episode_rng = random.Random(f"{self.seed}-{request_id}")  # the same for every turn of an episode
        text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        blocks = _split_chatml(text)
        system = blocks[0][1] if blocks and blocks[0][0] == "system" else ""
        if "Write the merged skill now." in text:
            reply = self._merge(blocks[-1][1] if blocks else "")
        else:
            reply = self._act(system, blocks, rng, ignores=episode_rng.random() < IGNORE_LOADED)
        ids = self.tokenizer.encode(reply, add_special_tokens=False)[
            : max(1, sampling_params.get("max_tokens", 256) - 1)
        ]
        ids.append(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return TokenOutput(
            token_ids=ids,
            log_probs=[NEUTRAL] * len(ids),
            stop_reason="completed",
            num_preempted=0,
            extra_fields={"min_global_steps": 0, "max_global_steps": 0},
        )

    # ------------------------------------------------------------------ acting

    def _act(self, system: str, blocks: list[tuple[str, str]], rng: random.Random, ignores: bool = False) -> str:
        users = [c for r, c in blocks if r == "user"]
        actions = [_last_action(c) for r, c in blocks if r == "assistant"]
        goal = re.search(r"Your task is to:\s*(.+)", users[0])
        family, _, _ = parse_goal(goal.group(1) if goal else "")
        last_obs = users[-1]
        can_write = "Action: write_skill" in system and sum(a == "write_skill" for a in actions) < 2
        wrote = "write_skill" in actions

        # ---- the task is completed: one more turn, to write a skill or not
        if "Task completed." in last_obs:
            roll = rng.random()
            if roll < WRITE_GOOD and family in GOOD_SKILLS and not wrote:
                return _write_call(*_good_skill(family, rng))
            if roll < WRITE_GOOD + WRITE_VAGUE:
                return _write_call(*VAGUE_SKILL)
            if roll < WRITE_GOOD + WRITE_VAGUE + WRITE_MALFORMED:
                return "Thought: save it\nAction: write_skill\nRemember that drawers can be closed."
            return "Thought: nothing to save.\nAction: done"

        # ---- load a skill: a noisy choice at the first turn, or a second thought after a rejected command
        entries = re.findall(r"^\[(S\d+)\] (.+)$", system, flags=re.MULTILINE)
        tried_loading = any(a and a.startswith("load_skill") for a in actions)
        if entries and not tried_loading:
            matching = [h for h, line in entries if any(cue in line.lower() for cue in FAMILY_CUES.get(family, []))]
            if not actions:
                roll = rng.random()
                if roll < LOAD_MATCHING and matching:
                    return f"Thought: this skill is about my task.\nAction: load_skill {rng.choice(matching)}"
                if roll < LOAD_MATCHING + LOAD_RANDOM:
                    return f"Thought: maybe this helps.\nAction: load_skill {rng.choice(entries)[0]}"
            elif "Nothing happens" in last_obs and matching and rng.random() < LOAD_WHEN_STUCK:
                return f"Thought: I am stuck, a skill may help.\nAction: load_skill {rng.choice(matching)}"

        # ---- write a skill mid-episode, right after the appliance step worked
        last_action = actions[-1] if actions else None
        if (
            can_write
            and not wrote
            and family in GOOD_SKILLS
            and last_action
            and last_action.split(" ")[0] in ("heat", "cool", "clean")
            and last_obs.startswith("You ")
            and rng.random() < WRITE_MID_EPISODE
        ):
            return _write_call(*_good_skill(family, rng), thought="the appliance step worked, save the procedure")

        skills = "" if ignores else _bodies(users)
        detour_prob = 0.03 if skills else 0.15
        if rng.random() < detour_prob:
            receptacles = re.findall(r"\ba (\w+ \d+)", users[0].split("Your task is to")[0])
            if receptacles:
                return f"Thought: Let me look somewhere else.\nAction: go to {rng.choice(receptacles)}"
        return self._task_action(users, actions, skills)

    def _task_action(self, users: list[str], actions: list[Optional[str]], skills: str) -> str:
        """The next environment command: deterministic given the dialogue and the skills it knows."""
        first_obs = users[0]
        goal = re.search(r"Your task is to:\s*(.+)", first_obs)
        family, obj, target_type = parse_goal(goal.group(1) if goal else "")
        receptacles = re.findall(r"\ba (\w+ \d+)", first_obs.split("Your task is to")[0])
        # search open surfaces before the (many) cabinets and drawers, as anyone would
        receptacles.sort(key=lambda r: (SEARCH_PRIORITY.get(r.rsplit(" ", 1)[0], 5), int(r.rsplit(" ", 1)[1])))
        knows_move = "move <object> to" in skills
        knows_microwave = "microwave" in skills
        knows_lamp = "use <lamp>" in skills
        tried = {a for a in actions if a}

        # ---- replay the dialogue to recover the state
        location, holding, processed, placed = None, None, False, 0
        visited, closed, seen, failed = [], set(), {}, set()
        for action, obs in zip(actions, users[1:], strict=False):
            if action is None:
                continue
            if "Nothing happens" in obs:
                failed.add(action)
                continue
            if action.startswith("go to "):
                location = action[6:]
                visited.append(location)
                if "is closed" in obs:
                    closed.add(location)
                else:
                    seen[location] = re.findall(r"\ba (\w+ \d+)", obs)
            elif action.startswith("open "):
                closed.discard(action[5:])
                seen[action[5:]] = re.findall(r"\ba (\w+ \d+)", obs)
            elif action.startswith("take ") and "You pick up" in obs:
                holding = action[5:].split(" from ")[0]
            elif action.startswith("move ") and "You move" in obs:
                holding, placed = None, placed + 1
                processed = False
            elif action.split(" ")[0] in ("heat", "cool", "clean") and "You " in obs:
                processed = True

        def say(thought: str, command: str) -> str:
            return f"Thought: {thought}\nAction: {command}"

        def explore() -> str:
            for r in receptacles:
                # a receptacle the game refused to go to (it shares a spot with another one) is skipped
                if r not in visited and f"go to {r}" not in failed:
                    return say(f"I have not checked {r} yet.", f"go to {r}")
            return say("I have checked everything, look again.", "look")

        # ---- find and take the object
        want = obj if family != "pick2" or placed < 2 else ""
        if holding is None:
            if location in closed:
                return say(f"{location} is closed, open it.", f"open {location}")
            here = [o for o in seen.get(location, []) if o.rsplit(" ", 1)[0] == want]
            # for pick2 the second object must not be taken back out of the target receptacle
            if here and not (family == "pick2" and location.rsplit(" ", 1)[0] == target_type):
                return say(f"I see {here[0]}, take it.", f"take {here[0]} from {location}")
            return explore()

        # ---- task specific middle step
        if family == "look":
            lamps = [o for objs in seen.values() for o in objs if "lamp" in o]
            here = [o for o in seen.get(location, []) if "lamp" in o]
            if here:
                if not knows_lamp:
                    for guess in (f"turn on {here[0]}", f"examine {holding} with {here[0]}"):
                        if guess not in tried:
                            return say("Try the lamp.", guess)
                return say("Turn the lamp on while holding the object.", f"use {here[0]}")
            if lamps:
                lamp_loc = next(loc for loc, objs in seen.items() if lamps[0] in objs)
                return say("Go back to the lamp.", f"go to {lamp_loc}")
            return explore()
        if family in APPLIANCE and not processed:
            appliance, verb = APPLIANCE[family]
            stove = next((r for r in receptacles if r.startswith("stoveburner")), None)
            if family == "heat" and not knows_microwave and stove and f"heat {holding} with {stove}" not in tried:
                if location != stove:
                    return say("A stove should heat it.", f"go to {stove}")
                return say("Heat it on the stove.", f"heat {holding} with {stove}")
            if location != appliance:
                return say(f"Go to the {appliance}.", f"go to {appliance}")
            return say(f"Now {verb} it.", f"{verb} {holding} with {appliance}")

        # ---- place it
        target = next((r for r in receptacles if r.rsplit(" ", 1)[0] == target_type), None)
        if target is None:
            return explore()
        if location != target:
            return say(f"Carry it to {target}.", f"go to {target}")
        if target in closed:
            return say(f"{target} is closed, open it.", f"open {target}")
        if not knows_move:
            for guess in (f"put {holding} in {target}", f"put {holding} on {target}", f"place {holding} in {target}"):
                if guess not in tried:
                    return say("Put it down.", guess)
        return say("Place it with move.", f"move {holding} to {target}")

    # ------------------------------------------------------------------ merging

    def _merge(self, user: str) -> str:
        """Merged version of a cluster: the first skill's text under a new name."""
        m = re.search(r"### (.+)\nWhen to use: (.+)\n((?:.|\n)*?)(?:\n\nSkill \d+:|\n\nWrite the merged)", user)
        if not m:
            return "I cannot merge these."
        return f"Name: {m.group(1)} (merged)\nWhen to use: {m.group(2)}\nProcedure:\n{m.group(3).strip()}"

    # ------------------------------------------------------------------ log-probs of a given trajectory

    def _score(self, token_ids: list[int]) -> list[list[float]]:
        """``rows[t - 1] = [log-prob of token t]``, in the layout of verl's ``prompt_logprobs``.

        Every environment command of the trajectory is compared with what the policy would do in
        the context it is scored in. Skill actions, detours and anything else that depends on the
        policy's dice get the same value in every context.
        """
        tok = self.tokenizer
        start_id, end_id = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
        logprobs = [NEUTRAL] * len(token_ids)
        users: list[str] = []
        actions: list[Optional[str]] = []
        starts = [i for i, t in enumerate(token_ids) if t == start_id]
        for b, start in enumerate(starts):
            stop = starts[b + 1] if b + 1 < len(starts) else len(token_ids)
            end = next((i for i in range(start, stop) if token_ids[i] == end_id), stop - 1)
            role, _, content = tok.decode(token_ids[start + 1 : end], skip_special_tokens=False).partition("\n")
            content = re.sub(r"^<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
            if role.strip() == "user":
                users.append(content)
            elif role.strip() == "assistant" and content:
                action = _last_action(content)
                if action and users and not action.startswith(("load_skill", "write_skill", "done")):
                    expected = _last_action(self._task_action(users, actions, _bodies(users)))
                    logprobs[start : end + 1] = [LIKELY if action == expected else UNLIKELY] * (end + 1 - start)
                actions.append(action)
        return [[lp] for lp in logprobs[1:]] + [[0.0]]


def _split_chatml(text: str) -> list[tuple[str, str]]:
    blocks = []
    for part in text.split("<|im_start|>")[1:]:
        role, _, content = part.partition("\n")
        content = content.split("<|im_end|>")[0]
        content = re.sub(r"^<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        if role.strip() == "assistant" and not content.strip():
            continue  # the open generation prompt
        blocks.append((role.strip(), content.strip()))
    return blocks


def _last_action(reply: str) -> Optional[str]:
    matches = re.findall(r"action\s*:\s*(.+)", reply, flags=re.IGNORECASE)
    return matches[-1].strip().lower() if matches else None
