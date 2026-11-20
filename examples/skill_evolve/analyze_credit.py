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
"""Check the credit estimator against ground truth, on a rollout of the SCRIPTED policy.

With the scripted policy the true value of every written skill is known by construction:

- ``new``: a correct procedure for the task type, and the old library has no skill of that type.
  It fixes something the policy does wrong on its own, so its true value is positive;
- ``duplicate``: the same correct procedure, but the old library already has one. Its true
  incremental value is zero;
- ``vague``: a platitude that tells the policy nothing it acts on.

The script reads the unit dumps of a run and reports, per class, what the estimator said; how well
several estimators separate ``new`` from the rest (the method's usage-weighted credit, the same
with a load-only weight, the same against a baseline of the control episodes that used a library
skill themselves, and the plain difference of arm means); how often the admission rule
errs; whether the usage weighting is biased against an estimator that does not condition on usage;
and whether the advantage of the write_skill tokens orders writers by the value of what they wrote.

Usage::

    python examples/skill_evolve/analyze_credit.py --run_dir <OUT_DIR> [--out report.json]
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

VAGUE = "Be careful"
ESTIMATORS = ("credit", "credit_load_only", "credit_matched_baseline", "arm_difference")


def load_units(run_dir: str) -> list[dict]:
    units = []
    for path in sorted(glob.glob(os.path.join(run_dir, "trajectories", "groups_worker*.jsonl"))):
        with open(path) as f:
            units.extend(json.loads(line) for line in f if line.strip())
    return [u for u in units if u["kind"] == "train_unit"]


def arm_of(unit: dict, name: str) -> dict:
    return next(a for a in unit["arms"] if a["name"] == name)


def auc(positives: list[float], negatives: list[float]) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    units = load_units(args.run_dir)
    rows = []  # one per tested candidate
    write_pairs = {"with_task_term": [], "credit_only": []}
    for unit in units:
        n_checks = len(unit["checks"])
        library_names = {e["name"] for e in arm_of(unit, "check0/control")["index"]} if n_checks else set()
        classes = {}
        for cand in unit["candidates"]:
            sid = cand["skill_id"]
            if cand["name"] == VAGUE:
                kind = "vague"
            else:
                kind = "duplicate" if cand["name"] in library_names else "new"
            classes[sid] = kind
            episodes, matched = [], 0.0
            for j in range(n_checks):
                baseline = unit["baseline"][str(j)]
                # alternative baseline: the control episodes that really USED a library skill, weighted
                # by their own usage weight (falls back to the plain mean when hardly any did)
                control = arm_of(unit, f"check{j}/control")["episodes"]
                weights = [max((u["weight"] for u in e["usage"].values()), default=0.0) for e in control]
                if sum(weights) >= 1.0:
                    matched_baseline = sum(w * e["reward"] for w, e in zip(weights, control, strict=True)) / sum(
                        weights
                    )
                else:
                    matched_baseline = baseline
                for e in arm_of(unit, f"check{j}/{sid}")["episodes"]:
                    u = e["usage"][sid]
                    episodes.append((e["reward"] - baseline, u["weight"], float(u["loaded"])))
                    matched += u["weight"] * (e["reward"] - matched_baseline)
            gains = [g for g, _, _ in episodes]
            weighted = sum(g * w for g, w, _ in episodes)
            sum_w = sum(w for _, w, _ in episodes)
            loaded_gain = sum(g * ld for g, _, ld in episodes)
            sum_l = sum(ld for _, _, ld in episodes)
            rows.append(
                {
                    "step": unit["step"],
                    "family": unit["family"],
                    "class": kind,
                    "credit": cand["credit"],
                    "credit_load_only": loaded_gain / (sum_l + args.kappa),
                    "credit_matched_baseline": matched / (sum_w + args.kappa),
                    "arm_difference": mean(gains),
                    "n": len(episodes),
                    "loaded": sum_l,
                    "sum_weight": sum_w,
                    "weighted_gain": weighted,
                    "total_gain": sum(gains),
                }
            )

        # advantage of the write_skill tokens: does it order the writers of one group by what they wrote?
        source = arm_of(unit, "source")
        task_scores = [e["reward"] for e in source["episodes"]]
        mu = mean(task_scores)
        sd = statistics.stdev(task_scores) if len(task_scores) > 1 else 0.0
        writers = []
        for e in source["episodes"]:
            for w in e["writes"]:
                if w["candidate"] is None:
                    continue
                task_adv = (e["reward"] - mu) / (sd + 1e-6)
                writers.append((classes[w["candidate"]], w["write_advantage"] + task_adv, w["write_advantage"]))
        for kind_a, full_a, credit_a in writers:
            for kind_b, full_b, credit_b in writers:
                if kind_a == "new" and kind_b in ("vague", "duplicate"):
                    write_pairs["with_task_term"].append((full_a > full_b) + 0.5 * (full_a == full_b))
                    write_pairs["credit_only"].append((credit_a > credit_b) + 0.5 * (credit_a == credit_b))

    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class"]].append(r)

    report = {"units": len(units), "candidates": len(rows), "per_class": {}}
    for kind in ("new", "duplicate", "vague"):
        rs = by_class.get(kind, [])
        if not rs:
            continue
        credits = [r["credit"] for r in rs]
        report["per_class"][kind] = {
            "n": len(rs),
            "credit_mean": round(mean(credits), 4),
            "credit_std": round(statistics.pstdev(credits), 4) if len(credits) > 1 else 0.0,
            "credit_positive_frac": round(mean([c > 0 for c in credits]), 3),
            "loaded_frac": round(sum(r["loaded"] for r in rs) / sum(r["n"] for r in rs), 3),
            "usage_weight_per_load": round(sum(r["sum_weight"] for r in rs) / max(sum(r["loaded"] for r in rs), 1), 3),
            "arm_difference_mean": round(mean([r["arm_difference"] for r in rs]), 4),
            "arm_difference_positive_frac": round(mean([r["arm_difference"] > 0 for r in rs]), 3),
            "credit_matched_baseline_mean": round(mean([r["credit_matched_baseline"] for r in rs]), 4),
            "credit_matched_baseline_positive_frac": round(mean([r["credit_matched_baseline"] > 0 for r in rs]), 3),
        }

    positives = by_class.get("new", [])
    negatives = by_class.get("duplicate", []) + by_class.get("vague", [])
    report["separation_auc_new_vs_rest"] = {
        key: round(auc([r[key] for r in positives], [r[key] for r in negatives]), 3) for key in ESTIMATORS
    }
    report["separation_auc_new_vs_duplicate"] = {
        key: round(auc([r[key] for r in positives], [r[key] for r in by_class.get("duplicate", [])]), 3)
        for key in ESTIMATORS
    }
    report["admission_rule_credit_above_zero"] = {
        "new_rejected_frac": round(mean([r["credit"] <= 0 for r in positives]), 3),
        "rest_admitted_frac": round(mean([r["credit"] > 0 for r in negatives]), 3),
    }

    # Bias of the usage weighting. Pooled over all `new` candidates:
    #   effect_on_used = sum u*(r - baseline) / sum u          what the credit converges to (no shrinkage)
    #   wald           = sum (r - baseline) / sum u            the whole arm's gain attributed to the users
    # The second does not pick episodes by an outcome-dependent weight: it is unbiased when episodes
    # that did not use the skill gained nothing. A gap between the two is selection bias.
    if positives:
        total_w = sum(r["sum_weight"] for r in positives)
        report["usage_weighting_bias_new"] = {
            "effect_on_used": round(sum(r["weighted_gain"] for r in positives) / total_w, 4) if total_w else None,
            "wald_all_gain_over_users": round(sum(r["total_gain"] for r in positives) / total_w, 4)
            if total_w
            else None,
            "gain_of_episodes_with_zero_weight": round(
                (sum(r["total_gain"] for r in positives) - sum(r["weighted_gain"] for r in positives))
                / max(sum(r["n"] for r in positives) - total_w, 1e-9),
                4,
            ),
        }

    report["write_advantage_orders_new_above_rest"] = {
        key: {"pairs": len(v), "frac": round(mean(v), 3) if v else None} for key, v in write_pairs.items()
    }
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"report": report, "candidates": rows}, f, indent=1)


if __name__ == "__main__":
    main()
