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
"""The credit estimator: usage weights, the credit formula and the reward of write_skill calls."""

import pytest

from verl.experimental.skill_loop.credit import (
    Sample,
    credit,
    group_normalize,
    influence,
    pooled_credit,
    usage_weight,
    write_advantages,
)


def test_usage_weight():
    assert usage_weight(False, 5.0, tau=0.1) == 0.0  # not loaded
    assert usage_weight(True, None, tau=0.1) == 0.0  # loaded with the last action: nothing to measure
    assert usage_weight(True, 0.0, tau=0.1) == 0.0  # loaded and ignored
    assert usage_weight(True, -0.3, tau=0.1) == 0.0
    assert usage_weight(True, 0.05, tau=0.1) == pytest.approx(0.5)
    assert usage_weight(True, 0.4, tau=0.1) == 1.0
    assert usage_weight(True, 0.0, tau=0.0) == 1.0  # gate disabled


def test_credit_worked_example():
    # two check tasks (baselines 0.4 and 0.5), two episodes each, kappa = 1
    samples = [
        Sample(score=1.0, baseline=0.4, weight=1.0, loaded=True),
        Sample(score=0.6, baseline=0.4, weight=0.0, loaded=False),  # never loaded: does not count
        Sample(score=0.3, baseline=0.5, weight=0.1, loaded=True),  # loaded, hardly followed
        Sample(score=0.9, baseline=0.5, weight=1.0, loaded=True),
    ]
    result = credit(samples, kappa=1.0)
    assert result.numerator == pytest.approx(0.98) and result.weight == pytest.approx(2.1)
    assert result.value == pytest.approx(0.98 / 3.1) and (result.n, result.n_loaded) == (4, 3)


def test_credit_cases():
    def one(score, weight, n=4):
        return credit([Sample(score, 0.5, weight, weight > 0)] * n, kappa=1.0).value

    assert one(1.0, 0.0) == 0.0  # nobody used it
    assert one(0.5, 1.0) == 0.0  # used, no gain
    assert one(0.9, 1.0) > 0 > one(0.1, 1.0)
    # the same gain counts for less when fewer episodes used the skill
    assert one(0.9, 1.0, n=1) < one(0.9, 1.0, n=8)
    # a skill that only helps on one of two check tasks is pulled down
    both = credit([Sample(0.9, 0.5, 1.0)] * 4, 1.0).value
    one_task = credit([Sample(0.9, 0.5, 1.0)] * 2 + [Sample(0.5, 0.5, 1.0)] * 2, 1.0).value
    assert one_task < both


def test_pooled_credit_is_the_credit_of_the_union():
    a = [Sample(0.9, 0.5, 1.0), Sample(0.2, 0.5, 0.5)]
    b = [Sample(0.7, 0.4, 1.0)]
    ca, cb = credit(a, 1.0), credit(b, 1.0)
    assert pooled_credit([(ca.numerator, ca.weight), (cb.numerator, cb.weight)], 1.0) == pytest.approx(
        credit(a + b, 1.0).value
    )


def test_influence_reads_shifted_positions_after_the_load():
    # tokens: 0-1 prompt, 2-3 generated before the load, 4-6 load observation, 7-8 generated after
    generated = [0, 0, 1, 1, 0, 0, 0, 1, 1]
    with_body = [0, 0, -1, -1, 0, 0, 0, -0.1, -0.3]
    # the blanked observation is one token shorter: positions after it move by -1
    without_body = [0, 0, -1, -1, 0, 0, -2.1, -0.3]
    assert influence(with_body, without_body, generated, segment_end=7, shift=-1) == pytest.approx(1.0)
    assert influence(with_body, without_body, [0] * 9, segment_end=7, shift=-1) is None


def test_group_normalize():
    assert group_normalize([0.5]) == [0.0]
    assert group_normalize([1.0, 1.0, 1.0]) == pytest.approx([0.0, 0.0, 0.0])
    z = group_normalize([0.0, 1.0])
    assert z == pytest.approx([-0.7071, 0.7071], abs=1e-3)
    assert group_normalize([0.0, 1.0], by_std=False) == [-0.5, 0.5]


def test_write_advantages_count_non_writers_as_zero_credit():
    # 4 source episodes: two wrote (credits +0.3 and -0.1), two did not
    adv = write_advantages([[0.3], [-0.1], [], []], by_std=False)
    assert adv[0] == pytest.approx([0.25]) and adv[1] == pytest.approx([-0.15]) and adv[2] == adv[3] == []
    # a single writer still has a comparison: its skill against "no skill"
    assert write_advantages([[0.2], [], [], []], by_std=False)[0] == pytest.approx([0.15])
    assert write_advantages([[-0.2], [], [], []])[0][0] < 0 < write_advantages([[0.2], [], [], []])[0][0]
    # nothing to compare when every credit is the same
    assert write_advantages([[0.0], [0.0], []]) == [[0.0], [0.0], []]


def test_write_advantages_with_several_and_untested_skills():
    # an episode may write two skills, each with its own credit
    adv = write_advantages([[0.4, 0.0], []], by_std=False)
    assert adv[0] == pytest.approx([0.4 - 0.4 / 3, -0.4 / 3])
    # an untested skill (credit None) gets no signal and does not enter the group
    adv = write_advantages([[None], [0.3], []], by_std=False)
    assert adv[0] == [0.0] and adv[1] == pytest.approx([0.15])
