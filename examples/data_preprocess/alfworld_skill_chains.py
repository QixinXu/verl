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
"""Build the ALFWorld *skill-chain* dataset used by ``verl.experimental.skill_loop``.

A training row is not a task but a unit ``(A; B_1..B_m)`` of one task type:

- ``source`` (A): the task on which the policy may write new skills;
- ``probes`` (B_j): *different* task instances of the same task type on which a new skill is
  checked. The credit measured there is the reward of the ``write_skill`` call.

What the script decides, and why (see docs/skill_evolve/DESIGN.md, section 1):

- **Type split.** ``cool`` and ``pick2`` never appear in training rows, neither as source nor as
  check task. ``cool`` is the same procedure as ``heat`` with another appliance (analogical
  transfer), ``pick2`` is ``pick`` done twice (compositional transfer): they measure whether
  skills transfer.
- **Check tasks.** Same task type as the source, with a different object *and* a different room
  where possible, and never the source's (object, receptacle, room) triple: the procedure carries
  over, the answer does not, so a skill cannot pay off by memorizing the source instance.
  (``--p_same`` below 1 mixes in check tasks of a related type; the method itself uses 1.)
- **Order.** Rows are written in curriculum order and must be read with ``data.shuffle=False``:
  phase 0 is the two easy types, phase 1 adds the composed types, phase 2 is uniform. Inside every
  step-sized block only ``families_per_block`` types appear and types rotate, so a skill admitted
  at step ``t`` is on offer for a matching task within the next one or two steps.

Usage::

    export ALFWORLD_DATA=/path/to/alfworld_data   # after `alfworld-download`
    python examples/data_preprocess/alfworld_skill_chains.py --local_save_dir ~/data/alfworld_skill
"""

import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict

import datasets

TASK_TYPES = {
    "pick_and_place_simple": "pick",
    "look_at_obj_in_light": "look",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_two_obj_and_place": "pick2",
}
TRAIN_FAMILIES = ["pick", "look", "clean", "heat"]
HELDOUT_FAMILIES = ["cool", "pick2"]
LEVEL = {"pick": 1, "look": 1, "clean": 2, "heat": 2, "cool": 2, "pick2": 3}
# source family -> related families that reuse one of its sub-procedures (only training families)
TRANSFER = {"pick": ["clean", "heat"], "look": ["pick"], "clean": ["heat", "pick"], "heat": ["clean", "pick"]}

GOAL_TEMPLATES = {
    "pick": "put a {obj} in {recep}",
    "look": "look at {obj} under the {toggle}",
    "clean": "clean some {obj} and put it in {recep}",
    "heat": "heat some {obj} and put it in {recep}",
    "cool": "cool some {obj} and put it in {recep}",
    "pick2": "put two {obj} in {recep}",
}


def load_games(data_root: str, split: str) -> list[dict]:
    games = []
    for gamefile in sorted(glob.glob(os.path.join(data_root, split, "*", "*", "game.tw-pddl"))):
        trial_dir = os.path.dirname(gamefile)
        with open(gamefile) as f:
            if not json.load(f).get("solvable", False):
                continue
        with open(os.path.join(trial_dir, "traj_data.json")) as f:
            traj = json.load(f)
        family = TASK_TYPES.get(traj["task_type"])
        if family is None:  # movable-receptacle tasks are not part of the six text task types
            continue
        params = traj["pddl_params"]
        obj, recep, toggle = (params[k].lower() for k in ("object_target", "parent_target", "toggle_target"))
        rel = os.path.relpath(gamefile, data_root)
        games.append(
            {
                "task_id": os.path.dirname(rel),
                "family": family,
                "payload": {
                    "gamefile": rel,
                    "goal": GOAL_TEMPLATES[family].format(obj=obj, recep=recep, toggle=toggle),
                    "object": obj,
                    "receptacle": recep or toggle,
                    "scene": traj["scene"]["floor_plan"],
                },
            }
        )
    return games


def pick_probes(source: dict, by_family: dict[str, list[dict]], rng: random.Random, k: int, p_same: float):
    """Return ``k`` probe tasks and the relation of the first one (``same`` | ``transfer``)."""
    probes, relations = [], []
    src = source["payload"]
    for _ in range(k):
        same = rng.random() < p_same
        family = source["family"] if same else rng.choice(TRANSFER[source["family"]])
        pool = by_family[family]
        # prefer a different object AND a different room; never the same (object, receptacle, room)
        strict = [
            g
            for g in pool
            if g["payload"]["object"] != src["object"] and g["payload"]["scene"] != src["scene"] and g not in probes
        ]
        loose = [
            g
            for g in pool
            if (g["payload"]["object"], g["payload"]["receptacle"], g["payload"]["scene"])
            != (src["object"], src["receptacle"], src["scene"])
            and g not in probes
        ]
        probes.append(rng.choice(strict or loose))
        relations.append("same" if same else "transfer")
    return probes, relations


def curriculum_order(
    rows_by_family: dict[str, list[dict]],
    block: int,
    families_per_block: int,
    phase0_blocks: int,
    phase1_blocks: int,
    rng: random.Random,
) -> list[dict]:
    """Interleave families block by block (one block = one training step).

    phase 0: level-1 families only; phase 1: all training families, type-blocked rotation;
    phase 2: all remaining rows, uniformly mixed.
    """
    queues = {f: list(rows) for f, rows in rows_by_family.items()}
    for q in queues.values():
        rng.shuffle(q)
    ordered: list[dict] = []
    per_family = max(1, block // families_per_block)

    def take_blocks(families: list[str], n_blocks: int, phase: int) -> None:
        cursor = 0
        for _ in range(n_blocks):
            chosen = [families[(cursor + j) % len(families)] for j in range(families_per_block)]
            cursor += families_per_block
            batch = []
            for fam in chosen:
                for _ in range(per_family):
                    if queues[fam]:
                        batch.append({**queues[fam].pop(), "phase": phase})
            ordered.extend(batch)

    take_blocks([f for f in TRAIN_FAMILIES if LEVEL[f] == 1], phase0_blocks, phase=0)
    take_blocks(TRAIN_FAMILIES, phase1_blocks, phase=1)
    rest = [{**row, "phase": 2} for q in queues.values() for row in q]
    rng.shuffle(rest)
    ordered.extend(rest)
    return ordered


def to_row(index: int, split: str, source: dict, probes: list[dict] | None = None, **extra) -> dict:
    extra_info = {
        "index": index,
        "split": split,
        "family": source["family"],
        "level": LEVEL[source["family"]],
        "source": source,
        **extra,
    }
    if probes is not None:
        extra_info["probes"] = probes
    return {
        "data_source": f"alfworld/{split}/{source['family']}",
        "agent_name": "skill_agent",
        "prompt": [{"role": "user", "content": source["payload"]["goal"]}],
        "extra_info": extra_info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alfworld_data", default=os.environ.get("ALFWORLD_DATA"))
    parser.add_argument("--local_save_dir", default="~/data/alfworld_skill")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_probes", type=int, default=2, help="check tasks stored per unit")
    parser.add_argument("--p_same", type=float, default=1.0, help="probability of a same-family check task")
    parser.add_argument("--block", type=int, default=8, help="rows per training step (data.train_batch_size)")
    parser.add_argument("--families_per_block", type=int, default=2)
    parser.add_argument("--phase0_blocks", type=int, default=10)
    parser.add_argument("--phase1_blocks", type=int, default=40)
    parser.add_argument(
        "--heldout_from_train",
        type=int,
        default=100,
        help="extra held-out-type games per family taken from the (unused) train split",
    )
    parser.add_argument("--val_per_family", type=int, default=0, help="cap per family and split (0 = all)")
    args = parser.parse_args()

    assert args.alfworld_data, "set ALFWORLD_DATA or pass --alfworld_data"
    data_root = os.path.join(args.alfworld_data, "json_2.1.1")
    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    rng = random.Random(args.seed)

    train_games = load_games(data_root, "train")
    by_family = defaultdict(list)
    for g in train_games:
        by_family[g["family"]].append(g)

    # ---- train chains: one per source game of a training family
    rows_by_family = defaultdict(list)
    for family in TRAIN_FAMILIES:
        for source in by_family[family]:
            probes, relations = pick_probes(source, by_family, rng, args.num_probes, args.p_same)
            rows_by_family[family].append({"source": source, "probes": probes, "probe_relation": relations[0]})
    ordered = curriculum_order(
        rows_by_family, args.block, args.families_per_block, args.phase0_blocks, args.phase1_blocks, rng
    )
    train_rows = [
        to_row(i, "train", r["source"], r["probes"], probe_relation=r["probe_relation"], phase=r["phase"])
        for i, r in enumerate(ordered)
    ]

    # ---- evaluation sets (no probes: evaluation never writes skills)
    def cap(games: list[dict]) -> list[dict]:
        if not args.val_per_family:
            return games
        counts, kept = Counter(), []
        for g in games:
            if counts[g["family"]] < args.val_per_family:
                counts[g["family"]] += 1
                kept.append(g)
        return kept

    seen = load_games(data_root, "valid_seen")
    unseen = load_games(data_root, "valid_unseen")
    val_sets = {
        # training task types, rooms seen in training
        "val_seen": cap([g for g in seen if g["family"] in TRAIN_FAMILIES]),
        # training task types, unseen rooms
        "val_unseen": cap([g for g in unseen if g["family"] in TRAIN_FAMILIES]),
        # task types that never produced a gradient or a skill
        "val_heldout": cap([g for g in seen + unseen if g["family"] in HELDOUT_FAMILIES]),
    }
    extra_heldout = []
    for family in HELDOUT_FAMILIES:
        pool = list(by_family[family])
        rng.shuffle(pool)
        extra_heldout.extend(pool[: args.heldout_from_train])
    val_sets["test_heldout_large"] = extra_heldout

    datasets.Dataset.from_list(train_rows).to_parquet(os.path.join(save_dir, "train.parquet"))
    summary = {
        "args": {k: v for k, v in vars(args).items() if k != "alfworld_data"},
        "train": {
            "rows": len(train_rows),
            "families": dict(Counter(r["extra_info"]["family"] for r in train_rows)),
            "probe_relation": dict(Counter(r["extra_info"]["probe_relation"] for r in train_rows)),
            "phase": dict(Counter(r["extra_info"]["phase"] for r in train_rows)),
        }
    }
    for name, games in val_sets.items():
        rows = [to_row(i, name, g) for i, g in enumerate(games)]
        datasets.Dataset.from_list(rows).to_parquet(os.path.join(save_dir, f"{name}.parquet"))
        summary[name] = {"rows": len(rows), "families": dict(Counter(g["family"] for g in games))}
    # a small mixed validation file for in-training validation
    mixed = []
    for name in ("val_seen", "val_unseen", "val_heldout"):
        per_family = defaultdict(list)
        for g in val_sets[name]:
            per_family[g["family"]].append(g)
        for games in per_family.values():
            for g in games[:8]:
                mixed.append(to_row(len(mixed), name, g))
    datasets.Dataset.from_list(mixed).to_parquet(os.path.join(save_dir, "val_mixed_small.parquet"))
    summary["val_mixed_small"] = {"rows": len(mixed), "sources": dict(Counter(r["data_source"] for r in mixed))}

    first_blocks = [
        [r["extra_info"]["family"] for r in train_rows[b * args.block : (b + 1) * args.block]] for b in range(14)
    ]
    summary["first_blocks"] = ["".join(f[0] if f != "pick" else "P" for f in blk) for blk in first_blocks]
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
