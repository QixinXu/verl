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
"""Prompt rendering and reply parsing of the skill loop."""

from verl.experimental.skill_loop import prompts

INDEX = [
    {"handle": "S1", "name": "Heat with the microwave", "when_to_use": "The task is to heat an object", "text": "X"},
    {"handle": "S2", "name": " ".join(["long"] * 40), "when_to_use": " ".join(["when"] * 80), "text": "Y"},
]


def test_without_an_index_only_write_skill_is_offered():
    system = prompts.build_solve_messages("obs", "GUIDE")[0]["content"]
    assert "## Skills" in system and "load_skill" not in system and "Action: write_skill" in system
    assert system.rstrip().endswith("Action: <one command>")


def test_index_shows_only_name_and_when_to_use_clipped():
    system = prompts.build_solve_messages("obs", "GUIDE", index=INDEX)[0]["content"]
    assert "[S1] Heat with the microwave - The task is to heat an object" in system
    assert "load_skill S<number>" in system
    # the body is not in the prompt, and neither field can be used to smuggle it in
    assert "\nX" not in system and "\nY" not in system
    line = next(row for row in system.splitlines() if row.startswith("[S2]"))
    assert len(line.split()) <= 1 + prompts.INDEX_NAME_WORDS + 1 + prompts.INDEX_WHEN_WORDS


def test_write_skill_is_offered_on_every_task():
    system = prompts.build_solve_messages("obs", "GUIDE", max_skill_words=77)[0]["content"]
    assert "Action: write_skill" in system and "at most 77 words" in system and "load_skill" not in system
    both = prompts.build_solve_messages("obs", "GUIDE", index=INDEX)[0]["content"]
    assert "load_skill" in both and "Action: write_skill" in both


def test_loaded_body_and_its_blank_counterfactual_differ_only_in_the_body():
    entry = {"handle": "S3", "text": "### Name\n1. do it"}
    real, blank = prompts.render_loaded(entry), prompts.render_loaded(entry, blank=True)
    assert real == "Loaded skill S3:\n### Name\n1. do it\n\nContinue with the task."
    assert blank == real.replace("### Name\n1. do it", "")


def test_parse_action_and_load():
    assert prompts.parse_action("Thought: x\nAction: Go To Fridge 1.") == "go to fridge 1"
    assert prompts.parse_action("no action here") is None
    for command in ("load_skill s3", "load_skill(s3)", "load_skill [s3]", "load skill s3", "load s3"):
        assert prompts.parse_load(command) == "s3"
    assert prompts.parse_load("load_skill") == ""
    assert prompts.parse_load("go to fridge 1") is None and prompts.parse_load("unload s3") is None


def test_write_call_is_split_from_the_reply_and_parsed():
    reply = (
        "Thought: worth saving\nAction: write_skill\nName: Heat it\nWhen to use: heating tasks\n"
        "Procedure:\n1. take it\n2. heat it with the microwave"
    )
    skill = prompts.parse_skill(prompts.split_write(reply), max_words=120)
    assert skill == {
        "name": "Heat it",
        "when_to_use": "heating tasks",
        "procedure": "1. take it\n2. heat it with the microwave",
    }
    assert prompts.split_write("Thought: x\nAction: go to fridge 1") is None
    assert prompts.split_write("Thought: x\nAction: write skill\nName: a") == "Name: a"
    # a missing part is not a skill; the body is truncated, so length buys nothing
    assert prompts.parse_skill("Name: a\nProcedure:\n1. x", 120) is None
    assert prompts.parse_skill("Name: a\nWhen to use: b\nProcedure:\n", 120) is None
    long = prompts.parse_skill("Name: a\nWhen to use: b\nProcedure:\n" + " ".join(["w"] * 300), 50)
    assert len(long["procedure"].split()) == 50
    # a reply that runs on into the next turn keeps only the procedure
    tail = prompts.parse_skill("Name: a\nWhen to use: b\nProcedure:\n1. x\n2. y\nAction: done", 120)
    assert tail["procedure"] == "1. x\n2. y"


def test_merge_prompt_lists_the_cluster():
    messages = prompts.build_merge_messages([{"text": "### A\n1. x"}, {"text": "### B\n1. y"}], 60)
    assert "at most 60 words" in messages[0]["content"]
    assert "Skill 1:\n### A" in messages[1]["content"] and "Skill 2:\n### B" in messages[1]["content"]
