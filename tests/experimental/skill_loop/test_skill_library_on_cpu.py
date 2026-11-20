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
"""The skill library: retrieval, admission, re-tests, merges, capacity, persistence."""

import pytest

from verl.experimental.skill_loop.embedding import HashEmbedder, cosine
from verl.experimental.skill_loop.skill_library import SkillLibrary


def _cand(name, when, procedure, credit):
    return {"name": name, "when_to_use": when, "procedure": procedure, "credit": credit}


HEAT = ("Heat with the microwave", "The task is to heat an object and put it somewhere", "1. take it\n2. microwave")
CLEAN = ("Clean at the sink", "The task is to clean an object and put it somewhere", "1. take it\n2. sinkbasin")
LAMP = ("Examine under a lamp", "The task is to look at an object under a lamp", "1. take it\n2. use the lamp")


def _library(**config):
    return SkillLibrary({"embedder": "hash", **config})


def _admit(lib, spec, credit=0.3, family="heat", task="t0", step=1):
    lib.propose({"family": family, "source_task_id": task, "step": step, "candidates": [_cand(*spec, credit)]})
    lib.commit(step)
    return next(reversed(lib.skills.values()))


def test_hash_embedding_is_unit_length_and_sees_shared_words():
    a, b, c = HashEmbedder().encode(["heat some egg", "heat an egg and put it away", "look at the lamp"])
    assert cosine(a, a) == pytest.approx(1.0) and cosine(a, b) > cosine(a, c) == 0.0


def test_retrieve_takes_the_most_similar_skills_and_orders_them_by_utility():
    lib = _library(index_size=2)
    heat = _admit(lib, HEAT, credit=0.1, family="heat", task="t1")
    clean = _admit(lib, CLEAN, credit=0.4, family="clean", task="t2")
    lamp = _admit(lib, LAMP, credit=0.9, family="look", task="t3")
    got = lib.retrieve("heat some egg and put it in fridge")
    # the lamp skill has the highest utility but is not about this task: it is not recalled;
    # among the recalled ones the order is by utility
    assert [e["skill_id"] for e in got] == [clean.skill_id, heat.skill_id]
    assert got[1]["similarity"] > got[0]["similarity"] and lamp.skill_id not in [e["skill_id"] for e in got]
    assert set(got[0]) >= {"skill_id", "name", "when_to_use", "text", "family", "utility", "similarity"}
    assert got[1]["text"].startswith("### Heat with the microwave\nWhen to use:")
    assert [e["skill_id"] for e in lib.retrieve("heat some egg", n=1)] == [heat.skill_id]


def test_retrieve_never_returns_a_skill_to_the_task_it_was_written_on():
    lib = _library()
    heat = _admit(lib, HEAT, task="t1")
    assert lib.retrieve("heat some egg", task_id="t1") == []
    assert [e["skill_id"] for e in lib.retrieve("heat some egg", task_id="t2")] == [heat.skill_id]
    assert lib.retrieve("heat some egg", task_id="t2", exclude_ids=(heat.skill_id,)) == []


def test_rollouts_only_see_committed_skills():
    lib = _library()
    lib.propose({"family": "heat", "source_task_id": "t1", "step": 1, "candidates": [_cand(*HEAT, 0.3)]})
    assert lib.retrieve("heat some egg") == [] and lib.version == 0
    lib.commit(1)
    assert len(lib.retrieve("heat some egg")) == 1 and lib.version == 1


def test_only_positive_credit_is_admitted():
    lib = _library()
    lib.propose(
        {
            "family": "heat",
            "source_task_id": "t1",
            "step": 1,
            "candidates": [_cand(*HEAT, 0.2), _cand(*CLEAN, 0.0), _cand(*LAMP, -0.3)],
        }
    )
    summary = lib.commit(1)
    assert summary["admitted"] == 1 and summary["rejected_no_credit"] == 2 and summary["num_active"] == 1
    skill = next(iter(lib.skills.values()))
    assert (skill.utility, skill.created_step, skill.verified_step, skill.family) == (0.2, 1, 1, "heat")


def test_near_duplicate_keeps_only_the_one_with_the_higher_credit():
    lib = _library()
    first = _admit(lib, HEAT, credit=0.3, task="t1")
    # the same skill again with a lower credit: dropped
    lib.propose({"family": "heat", "source_task_id": "t2", "step": 2, "candidates": [_cand(*HEAT, 0.1)]})
    assert lib.commit(2)["duplicate_dropped"] == 1 and first.utility == 0.3
    # ... with a higher credit: its text and credit replace the old one, the id stays
    better = ("Heat it", HEAT[1], HEAT[2] + " 1")
    lib.propose({"family": "heat", "source_task_id": "t3", "step": 3, "candidates": [_cand(*better, 0.5)]})
    summary = lib.commit(3)
    assert summary["duplicate_replaced"] == 1 and summary["num_active"] == 1
    assert (first.name, first.utility, first.verified_step) == ("Heat it", 0.5, 3)
    # the merged skill must not be shown to any task it was written on
    assert first.distilled_from("t1") and first.distilled_from("t2") and first.distilled_from("t3")


def test_cluster_texts_groups_duplicates_under_their_first_occurrence():
    lib = _library()
    texts = ["heat it with the microwave", "clean it at the sink", "heat it with the microwave", "use the lamp"]
    assert lib.cluster_texts(texts) == [0, 1, 0, 3]


def test_retest_plan_picks_due_skills_of_the_family_once_per_step():
    lib = _library(retest_every=2, retest_per_unit=1)
    old = _admit(lib, HEAT, task="t1", step=1)
    newer = _admit(lib, ("Heat fast", "heating tasks in a kitchen", "1. go to the microwave first"), task="t2", step=3)
    _admit(lib, CLEAN, family="clean", task="t3", step=1)
    assert lib.retest_plan("heat", step=2) == []  # nothing is due yet
    # at step 5 both heat skills are due; the one verified longer ago goes first
    assert [e["skill_id"] for e in lib.retest_plan("heat", step=5)] == [old.skill_id]
    # a second unit of the same step gets the next one, never the same skill twice
    assert [e["skill_id"] for e in lib.retest_plan("heat", step=5)] == [newer.skill_id]
    assert lib.retest_plan("heat", step=5) == []
    lib.commit(5)
    assert [e["skill_id"] for e in lib.retest_plan("heat", step=6)] == [old.skill_id]


def test_retest_priority_prefers_often_loaded_and_badly_performing_skills():
    lib = _library(retest_every=2)
    a = _admit(lib, HEAT, task="t1", step=1)
    b = _admit(lib, ("Heat fast", "heating tasks in a kitchen", "1. go to the microwave first"), task="t2", step=1)
    lib.report_loads([{"skill_id": b.skill_id, "n": 6, "outcome_sum": -1.8}])
    lib.commit(2)
    assert (b.n_loads, b.loads_since_verified) == (6, 6) and b.mean_outcome() == pytest.approx(-0.3)
    assert [e["skill_id"] for e in lib.retest_plan("heat", step=3, k=2)] == [b.skill_id, a.skill_id]


def test_retest_updates_utility_and_two_bad_retests_in_a_row_delete():
    lib = _library(delete_credit=0.02, delete_strikes=2, kappa=1.0)
    skill = _admit(lib, HEAT, credit=0.3)

    def retest(step, numerator, weight):
        lib.report_retest({"skill_id": skill.skill_id, "numerator": numerator, "weight": weight})
        return lib.commit(step)

    retest(3, numerator=1.0, weight=4.0)
    assert skill.utility == pytest.approx(0.2) and (skill.verified_step, skill.n_tests, skill.strikes) == (3, 1, 0)
    # one re-test at about zero is a strike, not a deletion ...
    retest(5, numerator=0.0, weight=0.0)
    assert skill.status == "active" and skill.strikes == 1 and skill.utility == 0.0
    # ... and a good re-test clears it
    retest(7, numerator=2.0, weight=4.0)
    assert skill.strikes == 0
    retest(9, numerator=-1.0, weight=4.0)
    summary = retest(11, numerator=0.05, weight=4.0)
    assert skill.status == "deleted" and skill.removed_reason == "retest_not_positive"
    assert summary["deleted_retest"] == 1 and lib.retrieve("heat some egg") == []


def test_retests_of_one_step_are_pooled():
    lib = _library(kappa=1.0)
    skill = _admit(lib, HEAT)
    lib.report_retest({"skill_id": skill.skill_id, "numerator": 1.0, "weight": 2.0})
    lib.report_retest({"skill_id": skill.skill_id, "numerator": 0.5, "weight": 2.0})
    lib.commit(3)
    assert skill.utility == pytest.approx(1.5 / 5.0) and skill.n_tests == 1


def _two_similar(lib):
    a = _admit(lib, HEAT, task="t1")
    b = _admit(lib, (HEAT[0], HEAT[1], HEAT[2] + "\n3. then put it down"), task="t2")
    return a, b


def test_merge_plan_finds_a_cluster_on_merge_steps_only():
    lib = _library(merge_every=4, merge_threshold=0.5)
    a, b = _two_similar(lib)
    _admit(lib, CLEAN, family="clean", task="t3")
    assert len(lib.skills) == 3
    assert lib.merge_plan("heat", step=3) is None  # not a merge step
    assert lib.merge_plan("clean", step=4) is None  # a single skill is not a cluster
    assert [e["skill_id"] for e in lib.merge_plan("heat", step=4)] == [a.skill_id, b.skill_id]
    assert lib.merge_plan("heat", step=4) is None  # claimed for this step


def test_accepted_merge_replaces_the_cluster():
    lib = _library(merge_every=4, merge_threshold=0.5)
    a, b = _two_similar(lib)
    merged = {"name": "Heat (merged)", "when_to_use": HEAT[1], "procedure": "1. take\n2. microwave\n3. put"}
    lib.report_merge(
        {
            "member_ids": [a.skill_id, b.skill_id],
            "family": "heat",
            "merged": merged,
            "merged_credit": 0.3,
            "members_credit": 0.25,
        }
    )
    summary = lib.commit(4)
    assert summary["merged"] == 1 and summary["num_active"] == 1
    assert a.status == b.status == "merged" and a.removed_reason.startswith("merged_into:")
    new = next(s for s in lib.skills.values() if s.status == "active")
    assert new.name == "Heat (merged)" and new.utility == 0.3 and new.merged_from == [a.skill_id, b.skill_id]
    assert new.distilled_from("t1") and new.distilled_from("t2")


def test_rejected_merge_keeps_the_cluster_and_waits_before_trying_again():
    lib = _library(merge_every=4, merge_threshold=0.5, merge_cooldown=8)
    a, b = _two_similar(lib)
    for merged, credit in (({"name": "m", "when_to_use": "w", "procedure": "p"}, 0.1), (None, 0.0)):
        lib.report_merge(
            {
                "member_ids": [a.skill_id, b.skill_id],
                "family": "heat",
                "merged": merged,
                "merged_credit": credit,
                "members_credit": 0.25,
            }
        )
    summary = lib.commit(4)
    assert summary["merge_rejected"] == 2 and a.status == b.status == "active" and summary["num_active"] == 2
    assert lib.merge_plan("heat", step=8) is None and lib.merge_plan("heat", step=12) is not None


def test_capacity_evicts_the_lowest_utility_of_a_family():
    lib = _library(max_skills_per_family=2)
    specs = [(f"skill {w}", f"only for {w} tasks", f"1. do {w}") for w in ("alpha", "bravo", "charlie")]
    lib.propose(
        {
            "family": "heat",
            "source_task_id": "t1",
            "step": 1,
            "candidates": [_cand(*specs[0], 0.3), _cand(*specs[1], 0.1), _cand(*specs[2], 0.2)],
        }
    )
    lib.propose({"family": "clean", "source_task_id": "t2", "step": 1, "candidates": [_cand(*CLEAN, 0.05)]})
    summary = lib.commit(1)
    assert summary["deleted_capacity"] == 1 and summary["per_family"] == {"heat": 2, "clean": 1}
    assert {s.name for s in lib.skills.values() if s.status == "deleted"} == {"skill bravo"}


def test_save_and_load_round_trip(tmp_path):
    lib = _library()
    skill = _admit(lib, HEAT, credit=0.3)
    lib.report_loads([{"skill_id": skill.skill_id, "n": 3, "outcome_sum": 0.6}])
    lib.commit(2)
    path = lib.save(str(tmp_path / "lib.json"))
    other = _library()
    other.load(path)
    assert other.version == 2 and other.skills[skill.skill_id].n_loads == 3
    assert other.retrieve("heat some egg") == lib.retrieve("heat some egg")
    assert [c["step"] for c in other.commit_log] == [1, 2]
