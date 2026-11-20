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
"""Prompts and output parsing of the skill loop.

The policy has one reply format (``Thought: ... / Action: <command>``) and, next to the commands of
the environment, two skill actions it may use at any turn of an episode:

- ``load_skill S<k>``: put the body of an indexed skill into the context;
- ``write_skill`` followed by ``Name / When to use / Procedure``: propose a new skill.

Only the *index* of the retrieved skills (name and when-to-use) is in the prompt. A body is shown
after the policy loads it, as the observation of that action.
"""

from __future__ import annotations

import re
from typing import Any, Optional

SOLVE_SYSTEM = """You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

{action_guide}
{skills}
Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>"""  # noqa: E501

SKILLS_HEADER = """
## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.
"""  # noqa: E501

INDEX_BLOCK = """
Skills you can load (only the name and the "when to use" are listed, the procedure appears after loading):
{entries}
- load_skill S<number>: shows the procedure of that skill. Load a skill when its "when to use" matches your task.
"""  # noqa: E501

WRITE_BLOCK = """
- write_skill: saves what you learned in this task as a new skill for future tasks of the same kind. You may do this at any turn, also right after the task is completed. Reply like this:
Thought: <what is worth saving>
Action: write_skill
Name: <short title>
When to use: <one sentence describing the kind of task>
Procedure:
1. <step>
2. <step>
Write a general procedure: which steps in which order, which command works, how to recover from a typical mistake. It must not depend on this room, so never mention numbered objects such as "drawer 3". Future agents see only the name and the "when to use" line before they decide to load it. Use at most {max_words} words in the procedure.
"""  # noqa: E501

CONTINUE = "Continue with the task."
LOAD_UNKNOWN = "Nothing happens. There is no skill {handle} to load."
LOAD_AGAIN = "Skill {handle} is already loaded. " + CONTINUE
WRITE_SAVED = "Skill saved. " + CONTINUE
WRITE_MALFORMED = (
    "Skill not saved. After `Action: write_skill` give three parts:\n"
    "Name: <short title>\nWhen to use: <one sentence>\nProcedure:\n1. <step>"
)
TASK_COMPLETED = "Task completed. You may save a skill with write_skill, or reply `Action: done`."
FORMAT_ERROR = "Invalid reply. Use the format:\nThought: <one sentence>\nAction: <one command>"

INDEX_NAME_WORDS = 10
INDEX_WHEN_WORDS = 30

MERGE_SYSTEM = """You are given several skills that overlap. Merge them into ONE skill that keeps every useful step and drops repetition.

Reply in exactly this format (at most {max_words} words in the procedure):
Name: <short title>
When to use: <one sentence describing the kind of task>
Procedure:
1. <step>
2. <step>"""  # noqa: E501

MERGE_USER = "{skills}\n\nWrite the merged skill now."


def _clip_words(text: str, limit: int) -> str:
    words = (text or "").split()
    return " ".join(words[:limit]) + ("..." if len(words) > limit else "")


def render_index(index: list[dict[str, Any]]) -> str:
    """The retrieved skills as the policy sees them: one line per skill, name and when-to-use only.

    Both fields are clipped, so the size of the index is bounded by its number of entries and a
    skill writer cannot smuggle the procedure into the part that is always visible.
    """
    return "\n".join(
        f"[{e['handle']}] {_clip_words(e['name'], INDEX_NAME_WORDS)} - "
        f"{_clip_words(e['when_to_use'], INDEX_WHEN_WORDS)}"
        for e in index
    )


def build_solve_messages(
    first_observation: str,
    action_guide: str,
    index: Optional[list[dict[str, Any]]] = None,
    max_skill_words: int = 120,
) -> list[dict]:
    """System prompt and first observation of an episode.

    ``index``: retrieved skills, shown as name + when-to-use. ``write_skill`` is offered on every task.
    """
    skills = SKILLS_HEADER
    if index:
        skills += INDEX_BLOCK.format(entries=render_index(index))
    skills += WRITE_BLOCK.format(max_words=max_skill_words)
    system = SOLVE_SYSTEM.format(action_guide=action_guide, skills=skills)
    return [{"role": "system", "content": system}, {"role": "user", "content": first_observation}]


def render_loaded(entry: dict[str, Any], blank: bool = False) -> str:
    """Observation of ``load_skill``: the skill body.

    ``blank=True`` is the counterfactual used to measure influence: the same message with the body
    removed, so the two contexts differ in nothing but the body.
    """
    return f"Loaded skill {entry['handle']}:\n{'' if blank else entry['text']}\n\n{CONTINUE}"


_ACTION_RE = re.compile(r"action\s*:\s*(.+)", re.IGNORECASE)
_WRITE_RE = re.compile(r"action\s*:\s*`?write[_ ]skill\b[^\n]*\n?", re.IGNORECASE)
_LOAD_RE = re.compile(r"load(?:[_ ]skill)?\s*[\(\[]?\s*([a-z0-9_]*)\s*[\)\]]?")


def parse_action(text: str) -> Optional[str]:
    """Extract the command from a ``Thought: ... Action: ...`` reply (last ``Action:`` line wins)."""
    matches = _ACTION_RE.findall(text or "")
    if not matches:
        return None
    command = matches[-1].strip().splitlines()[0].strip()
    command = command.strip("`*\"' .").lower()
    command = re.sub(r"\s+", " ", command)
    return command or None


def parse_load(command: str) -> Optional[str]:
    """Handle named by a ``load_skill ...`` command (``"s3"`` or ``""``); None if it is not a load."""
    m = _LOAD_RE.fullmatch((command or "").strip().lower())
    return m.group(1) if m else None


def split_write(text: str) -> Optional[str]:
    """If the reply calls ``write_skill``, return what follows the call (the skill text), else None."""
    m = _WRITE_RE.search(text or "")
    return text[m.end() :] if m else None


def build_merge_messages(skills: list[dict[str, Any]], max_words: int) -> list[dict]:
    """Prompt that asks for one skill replacing a cluster of similar ones."""
    listing = "\n\n".join(f"Skill {i + 1}:\n{s['text']}" for i, s in enumerate(skills))
    return [
        {"role": "system", "content": MERGE_SYSTEM.format(max_words=max_words)},
        {"role": "user", "content": MERGE_USER.format(skills=listing)},
    ]


_NAME_RE = re.compile(r"^\s*name\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_WHEN_RE = re.compile(r"^\s*when to use\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_PROC_RE = re.compile(r"^\s*procedure\s*:\s*(.*)", re.IGNORECASE | re.MULTILINE | re.DOTALL)
_NEXT_TURN_RE = re.compile(r"^\s*(?:thought|action)\s*:", re.IGNORECASE | re.MULTILINE)


def parse_skill(text: str, max_words: int) -> Optional[dict[str, str]]:
    """Parse ``Name / When to use / Procedure``; None when a part is missing.

    The procedure is truncated to ``max_words``: writing more can never help, so length is not a
    lever the writer can pull.
    """
    text = (text or "").strip()
    name, when, proc = _NAME_RE.search(text), _WHEN_RE.search(text), _PROC_RE.search(text)
    if not (name and when and proc):
        return None
    # a reply may go on with the next turn (``Action: done``); that is not part of the skill
    procedure = _NEXT_TURN_RE.split(proc.group(1))[0].strip()
    if not procedure:
        return None
    if len(procedure.split()) > max_words:
        kept, count = [], 0
        for w in procedure.split(" "):
            count += len(w.split()) or 0
            if count > max_words:
                break
            kept.append(w)
        procedure = " ".join(kept).rstrip()
    return {
        "name": " ".join(name.group(1).split())[:80],
        "when_to_use": " ".join(when.group(1).split())[:240],
        "procedure": procedure,
    }


def render_skill(skill: dict[str, str]) -> str:
    """Same layout as ``Skill.render`` so a candidate looks exactly like a library skill once loaded."""
    return f"### {skill['name']}\nWhen to use: {skill['when_to_use']}\n{skill['procedure']}"
