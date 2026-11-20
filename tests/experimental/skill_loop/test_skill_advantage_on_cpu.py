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
"""Per-arm GRPO advantages, the token-level reward of write_skill calls and the step metrics."""

import pytest
import torch

from verl.experimental.skill_loop.skill_advantage import compute_skill_advantage, role_of
from verl.experimental.skill_loop.skill_trainer import _padded_like, _skill_metrics


def test_role_of():
    assert role_of("source") == "source" and role_of("eval") == "eval"
    assert role_of("check0/control") == "control" and role_of("check1/new2") == "candidate"
    assert role_of("check0/retest_sk0003/without") == "retest" and role_of("check0/merge/merged") == "merge"


def test_episodes_are_compared_only_inside_their_arm():
    scores = torch.tensor([1.0, 0.0, 0.9, 0.7, 0.2, 0.2, 0.5])
    uids = ["u1"] * 6 + ["u2"]
    groups = ["source", "source", "check0/control", "check0/control", "check0/new0", "check0/new0", "source"]
    mask = torch.ones(7, 3)
    adv = compute_skill_advantage(scores, uids, groups, mask)
    assert adv[0, 0] == pytest.approx(0.7071, abs=1e-3) and adv[1, 0] == pytest.approx(-0.7071, abs=1e-3)
    assert adv[2, 0] > 0 > adv[3, 0]
    # an arm whose episodes all scored the same carries no signal, whatever the other arms scored
    assert adv[4].abs().sum() == 0 and adv[5].abs().sum() == 0
    # an arm with a single episode has nothing to be compared with
    assert adv[6].abs().sum() == 0
    plain = compute_skill_advantage(scores, uids, groups, mask, norm_adv_by_std=False)
    assert plain[0, 0] == pytest.approx(0.5) and plain[2, 0] == pytest.approx(0.1)


def test_write_skill_tokens_get_task_advantage_plus_credit_advantage():
    scores = torch.tensor([1.0, 0.0])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    write_mask = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 0, 0, 0]])
    write_adv = torch.tensor([[0.0, 0.0, 0.8, 0.8, 0.0], [0.0] * 5])
    args = (scores, ["u"] * 2, ["source"] * 2, mask)
    adv = compute_skill_advantage(*args, write_mask=write_mask, write_adv=write_adv, norm_adv_by_std=False)
    # ordinary tokens: the task advantage; write_skill tokens: task advantage + normalized credit
    assert adv[0].tolist() == pytest.approx([0.5, 0.5, 1.3, 1.3, 0.0])
    assert adv[1].tolist() == pytest.approx([-0.5] * 5)
    # write_task_coef = 0: the write call is rewarded by the credit alone
    adv = compute_skill_advantage(
        *args, write_mask=write_mask, write_adv=write_adv, norm_adv_by_std=False, write_task_coef=0.0
    )
    assert adv[0].tolist() == pytest.approx([0.5, 0.5, 0.8, 0.8, 0.0])


def test_role_coefficient_scales_one_role():
    scores = torch.tensor([1.0, 0.0, 1.0, 0.0])
    groups = ["source", "source", "check0/control", "check0/control"]
    adv = compute_skill_advantage(
        scores, ["u"] * 4, groups, torch.ones(4, 1), norm_adv_by_std=False, role_coef={"control": 0.5}
    )
    assert adv.squeeze(-1).tolist() == pytest.approx([0.5, -0.5, 0.25, -0.25])


def test_token_columns_are_padded_to_the_response_shape():
    like = torch.zeros(2, 5)
    nested = torch.nested.as_nested_tensor([torch.tensor([1, 1, 0]), torch.tensor([0, 1])], layout=torch.jagged)
    assert _padded_like(nested, like).tolist() == [[1, 1, 0, 0, 0], [0, 1, 0, 0, 0]]
    assert _padded_like(torch.ones(2, 7), like).shape == (2, 5)


def _meta(arm, score, steps=5, n_loads=0, n_writes=0, check=-1, subject="", usage=None, credits=()):
    return {
        "role": arm,
        "arm": arm,
        "check": check,
        "subject": subject,
        "score": score,
        "won": float(score > 0),
        "steps": steps,
        "n_loads": n_loads,
        "n_writes": n_writes,
        "write_credits": list(credits),
        "usage": usage or {},
    }


def test_step_metrics():
    used = {"new0": {"loaded": True, "influence": 0.4, "weight": 1.0}}
    unused = {"new0": {"loaded": False, "influence": None, "weight": 0.0}}
    metas = [
        _meta("source", 1.0, n_writes=1, credits=[0.3]),
        _meta("source", 0.0),
        _meta("check0/control", 0.4, check=0),
        _meta("check0/control", 0.6, check=0),
        _meta("check0/new0", 0.9, check=0, subject="new0", n_loads=1, usage=used),
        _meta("check0/new0", 0.5, check=0, subject="new0", usage=unused),
        _meta("source", 9.9),  # padding row, ignored
    ]
    m = _skill_metrics(metas, ["u"] * 7, [False] * 6 + [True])
    assert m["skill/source_score_mean"] == pytest.approx(0.5) and m["skill/source_rows"] == 2
    assert m["skill/write_frac"] == pytest.approx(0.5) and m["skill/write_credit_mean"] == pytest.approx(0.3)
    assert m["skill/write_credit_pos_frac"] == 1.0
    assert m["skill/candidate_loaded_frac"] == pytest.approx(0.5)
    assert m["skill/candidate_usage_weight"] == 1.0 and m["skill/candidate_influence"] == pytest.approx(0.4)
    # plain lift of putting the new skill into the index: mean(0.9, 0.5) - mean(0.4, 0.6)
    assert m["skill/candidate_lift"] == pytest.approx(0.2)
    assert all(v == v for v in m.values())  # no NaN is logged
