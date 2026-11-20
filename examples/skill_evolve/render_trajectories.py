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
"""Turn the dumps of a skill-evolution run into a readable trajectory report.

Joins three artifacts written during rollout:

- ``trajectories/groups_worker*.jsonl``  one record per training unit (arms, messages, credits);
- ``trajectories/advantages.jsonl``      the advantages the trainer gave each episode;
- ``checkpoints/skill_library/step_*.json``  the library after each step's commit.

Usage::

    python examples/skill_evolve/render_trajectories.py --run_dir <OUT_DIR> --out report.md
"""

import argparse
import glob
import json
import os


def load(run_dir: str):
    units = []
    for path in sorted(glob.glob(os.path.join(run_dir, "trajectories", "groups_worker*.jsonl"))):
        with open(path) as f:
            units.extend(json.loads(line) for line in f if line.strip())
    advantages = {}
    adv_path = os.path.join(run_dir, "trajectories", "advantages.jsonl")
    if os.path.exists(adv_path):
        with open(adv_path) as f:
            for line in f:
                row = json.loads(line)
                advantages[row["key"]] = row
    libraries = {}
    for path in glob.glob(os.path.join(run_dir, "checkpoints", "skill_library", "step_*.json")):
        step = int(os.path.basename(path)[len("step_") : -len(".json")])
        with open(path) as f:
            libraries[step] = json.load(f)
    return units, advantages, libraries


def fmt(x, digits=3):
    return "-" if x is None else f"{x:+.{digits}f}"


def arm_of(unit: dict, name: str) -> dict:
    return next(a for a in unit["arms"] if a["name"] == name)


def adv_of(unit: dict, episode: dict, advantages: dict) -> dict:
    return advantages.get(f"{unit['uid']}_{episode['session_id']}_0", {})


def render_messages(messages: list[dict], max_chars: int = 1600) -> str:
    out = []
    for m in messages:
        content = m["content"]
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [{len(content) - max_chars} more characters]"
        out.append(f"[{m['role']}]\n{content}")
    return "\n\n".join(out)


def index_brief(arm: dict) -> str:
    return ", ".join(f"{e['handle']}={e['skill_id']}" for e in arm["index"]) or "(empty)"


def render_episode(unit: dict, arm: dict, episode: dict, advantages: dict, caption: str) -> list[str]:
    adv = adv_of(unit, episode, advantages)
    head = (
        f"**{caption}** — arm `{arm['name']}`, episode {episode['session_id']}: "
        f"{'won' if episode['won'] else 'failed'} in {episode['steps']} steps, reward {episode['reward']:.3f}, "
        f"advantage {fmt(adv.get('advantage'))}"
    )
    if adv.get("write_tokens"):
        head += f", advantage on its {adv['write_tokens']} write_skill tokens {fmt(adv.get('write_advantage'))}"
    return [head, "", "```text", render_messages(episode["messages"]), "```", ""]


def render_unit(unit: dict, advantages: dict, title: str, why: str) -> str:
    lines = [f"## {title}", "", why, ""]
    lines.append(
        f"Step {unit['step']}, task type `{unit['family']}`, library version {unit['library_version']} "
        f"(uid `{unit['uid']}`)."
    )
    lines += ["", f"- A (source task): `{unit['source']['goal']}`"]
    for j, check in enumerate(unit.get("checks", [])):
        lines.append(f"- B{j + 1} (check task): `{check['goal']}`, baseline score {unit['baseline'][str(j)]:.3f}")

    lines += [
        "",
        "### Arms",
        "",
        "| arm | task | skill index in the prompt | episodes | mean score |",
        "|---|---|---|---|---|",
    ]
    for arm in unit["arms"]:
        lines.append(
            f"| `{arm['name']}` | {arm['goal']} | {index_brief(arm)} | {len(arm['episodes'])} "
            f"| {arm['mean_score']:.3f} |"
        )

    candidates = unit.get("candidates") or []
    if candidates:
        lines += [
            "",
            "### New skills written on A and their credit",
            "",
            "| candidate | name | written by (episode, call) | loaded | sum of usage weights | credit |",
            "|---|---|---|---|---|---|",
        ]
        for c in candidates:
            lines.append(
                f"| {c['skill_id']} | {c['name']} | {c['writers']} | {c['n_loaded']}/{c['n']} | "
                f"{c['weight']:.2f} | {fmt(c['credit'])} |"
            )
        best = max(candidates, key=lambda c: abs(c["credit"]))
        lines += ["", f"**{best['skill_id']} — {best['name']}**", "", "```text"]
        lines += [f"When to use: {best['when_to_use']}\n{best['procedure']}", "```", ""]
        lines.append(
            f"Credit of `{best['skill_id']}`, episode by episode "
            "(credit = sum of u x (score - baseline) / (sum of u + kappa)):"
        )
        lines += ["", "| check task | episode | loaded | influence | u | score | baseline | u x (score - baseline) |"]
        lines.append("|---|---|---|---|---|---|---|---|")
        for j in range(len(unit["checks"])):
            arm = arm_of(unit, f"check{j}/{best['skill_id']}")
            baseline = unit["baseline"][str(j)]
            for e in arm["episodes"]:
                u = e["usage"][best["skill_id"]]
                lines.append(
                    f"| B{j + 1} | {e['session_id']} | {'yes' if u['loaded'] else 'no'} | {fmt(u['influence'])} | "
                    f"{u['weight']:.2f} | {e['reward']:.3f} | {baseline:.3f} | "
                    f"{fmt(u['weight'] * (e['reward'] - baseline))} |"
                )

    source = arm_of(unit, "source") if unit["kind"] == "train_unit" else None
    if source:
        lines += [
            "",
            "### Rewards of the source episodes",
            "",
            "Ordinary tokens get the task advantage. The tokens of a write_skill call get the task advantage "
            "plus the credit of the skill, normalized over the group (an episode that wrote nothing counts as "
            "credit 0).",
            "",
            "| episode | score of A | task advantage | wrote | credit | advantage of the write_skill tokens |",
            "|---|---|---|---|---|---|",
        ]
        for e in source["episodes"]:
            adv = adv_of(unit, e, advantages)
            wrote = ", ".join(f"{w['candidate'] or 'untested'}@step{w['step']}" for w in e["writes"]) or "-"
            credit = ", ".join(fmt(w["credit"]) for w in e["writes"]) or "-"
            lines.append(
                f"| {e['session_id']} | {e['reward']:.3f} | {fmt(adv.get('advantage'))} | {wrote} | {credit} | "
                f"{fmt(adv.get('write_advantage')) if e['writes'] else '-'} |"
            )

    for r in unit.get("retests") or []:
        lines += [
            "",
            f"**Re-test of library skill {r['skill_id']} ({r['name']})**: loaded in {r['n_loaded']}/{r['n']} "
            f"episodes, sum of usage weights {r['weight']:.2f}, credit {fmt(r['credit'])}.",
        ]
    merge = unit.get("merge")
    if merge:
        lines += [
            "",
            f"**Merge of {merge['member_ids']}**: credit of the merged skill {fmt(merge['merged_credit'])}, "
            f"credit of the original skills {fmt(merge['members_credit'])}.",
        ]

    lines += ["", "### Trajectories", ""]
    shown = []
    if source:
        writers = [e for e in source["episodes"] if e["writes"]]
        loaders = [e for e in source["episodes"] if e["loads"]]
        if writers:
            pick = max(writers, key=lambda e: max((w["credit"] or 0) for w in e["writes"]))
            shown.append((source, pick, "A source episode that wrote a skill"))
        elif loaders:
            shown.append((source, loaders[0], "A source episode that loaded a library skill"))
        if candidates:
            sid = best["skill_id"]
            arm = arm_of(unit, f"check0/{sid}")
            used = sorted(arm["episodes"], key=lambda e: -e["usage"][sid]["weight"])
            shown.append((arm, used[0], f"A check episode with the new skill {sid} in its index"))
            ignored = [e for e in arm["episodes"] if e["usage"][sid]["weight"] == 0]
            if ignored and ignored[0] is not used[0]:
                shown.append((arm, ignored[0], f"A check episode that did not use {sid} (usage weight 0)"))
        control = arm_of(unit, "check0/control") if unit.get("checks") else None
        if control:
            shown.append((control, control["episodes"][0], "A control episode (old library only)"))
    else:
        arm = unit["arms"][0]
        loaded = [e for e in arm["episodes"] if e["loads"]] or arm["episodes"]
        shown.append((arm, loaded[0], "A validation episode"))
    for arm, episode, caption in shown:
        lines += render_episode(unit, arm, episode, advantages, caption)
    return "\n".join(lines)


def pick_units(units: list[dict]) -> list[tuple[dict, str, str]]:
    """Choose a handful of units that together show writing, credit, loading, re-test, merge and rejection."""
    train = sorted((u for u in units if u["kind"] == "train_unit"), key=lambda u: u["step"])
    evals = [u for u in units if u["kind"] == "eval_group"]
    picked, used = [], set()

    def add(candidates, title, why):
        for u in candidates:
            if u["uid"] not in used:
                used.add(u["uid"])
                picked.append((u, title, why))
                return

    def best_credit(u):
        return max((c["credit"] for c in u.get("candidates") or []), default=-9)

    def source_index(u):
        return arm_of(u, "source")["index"]

    add(
        sorted((u for u in train if not source_index(u)), key=lambda u: (u["step"], -best_credit(u))),
        "Unit 1 — a new skill is written, checked and credited (empty library)",
        "Nothing can be loaded yet. Episodes on A write skills on their own initiative; every distinct skill is "
        "put into the index of the check tasks B1, B2, and its credit is what the episodes that really used it "
        "gained over the control episodes.",
    )
    add(
        sorted((u for u in train if source_index(u) and u.get("retests")), key=lambda u: -u["retests"][0]["credit"]),
        "Unit 2 — library skills are retrieved and loaded, and one is re-tested",
        "The index of A now lists retrieved library skills; loading one is the policy's own action. The same "
        "unit re-tests a library skill of its task type with the same credit estimator: library with the skill "
        "against library without it.",
    )
    add(
        sorted((u for u in train if u.get("candidates") and best_credit(u) <= 0), key=lambda u: u["step"]),
        "Unit 3 — skills that earn no credit and are not admitted",
        "No new skill of this unit has a positive credit: either the check episodes did not load it, or they "
        "loaded it without their actions depending on it, or using it did not beat the old library.",
    )
    add(
        sorted((u for u in train if u.get("merge")), key=lambda u: u["step"]),
        "Unit 4 — a merge of similar library skills is drafted and judged",
        "Similar skills of one task type are merged into one draft. The draft replaces them only if its credit "
        "is at least the credit of the skills it replaces, measured against the index without any of them.",
    )
    add(
        sorted((u for u in evals if u["arms"][0]["index"]), key=lambda u: -u["step"]),
        "Unit 5 — validation with the frozen library",
        "Validation only reads the library: skills are retrieved and may be loaded, nothing is written or staged.",
    )
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy_note", default="", help="one line describing what produced the tokens")
    parser.add_argument("--dump_selected", default="", help="also write the rendered units, raw, to this jsonl")
    args = parser.parse_args()

    units, advantages, libraries = load(args.run_dir)
    lines = ["# Skill-evolution rollout trajectories", ""]
    if args.policy_note:
        lines += [f"> {args.policy_note}", ""]
    lines.append(f"{len(units)} units (training and validation), {len(advantages)} episodes with advantages.")
    lines += [
        "",
        "## Library after each step",
        "",
        "| step | active skills | per task type | events |",
        "|---|---|---|---|",
    ]
    for step in sorted(libraries):
        log = [e for e in libraries[step]["commit_log"] if e["step"] == step]
        if not log:
            continue
        e = log[-1]
        events = "; ".join(f"{ev['event']} {ev.get('skill_id', ev.get('members', ''))}" for ev in e["events"]) or "-"
        lines.append(f"| {step} | {e['num_active']} | {json.dumps(e['per_family'])} | {events} |")
    if libraries:
        last = libraries[max(libraries)]
        lines += ["", f"### Skills in the library after step {max(libraries)}", ""]
        lines.append(
            "| id | status | task type | name | added | last verified | utility | loads | re-tests | strikes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in last["skills"]:
            status = s["status"] + (f" ({s['removed_reason']})" if s["removed_reason"] else "")
            lines.append(
                f"| {s['skill_id']} | {status} | {s['family']} | {s['name']} | {s['created_step']} | "
                f"{s['verified_step']} | {s['utility']:+.3f} | {s['n_loads']} | {s['n_tests']} | {s['strikes']} |"
            )
        for s in last["skills"]:
            if s["status"] != "active":
                continue
            header = f"**{s['skill_id']} — {s['name']}** (from `{s['source_task_id']}`, step {s['created_step']})"
            lines += ["", header, "", "```text", f"When to use: {s['when_to_use']}\n{s['procedure']}", "```"]

    selected = pick_units(units)
    for unit, title, why in selected:
        lines += ["", "---", "", render_unit(unit, advantages, title, why)]
    if args.dump_selected:
        with open(args.dump_selected, "w") as f:
            for unit, title, _ in selected:
                f.write(json.dumps({"title": title, "unit": unit}, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(selected)} units rendered")


if __name__ == "__main__":
    main()
