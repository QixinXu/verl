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
"""The global skill library: the persistent, cross-step state of skill-based self-evolution.

The library is not trained by gradient. It changes only at ``commit``, once per training step, so
every rollout of a step reads the same version (like it reads the same weights).

- **Read** (``retrieve``): coarse recall. The name and when-to-use of every skill are embedded;
  the ``index_size`` skills closest to the task description are returned, ordered by their current
  utility. Which of them to load is the policy's own action.
- **Plan** (``retest_plan`` / ``merge_plan``): which skills a rollout group should re-test or try to
  merge this step.
- **Stage** (``propose`` / ``report_loads`` / ``report_retest`` / ``report_merge``): rollouts append
  evidence. Every number that decides something is a *credit* from ``credit.py``: admission,
  re-test and merge all use the same estimator.
- **Commit**: apply the evidence. Admit candidates with positive credit, replace near-duplicates
  by the better one, delete skills whose re-tests stay at or below zero, swap in accepted merges,
  enforce the per-family capacity.

Pure Python apart from the optional Hugging Face embedder; the Ray actor wrapper is in
``skill_library_actor.py``.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from verl.experimental.skill_loop.credit import pooled_credit
from verl.experimental.skill_loop.embedding import cosine, make_embedder


@dataclass
class SkillLibraryConfig:
    # ---- retrieval ----
    embedder: str = "hash"
    """``hash`` (no model) or a Hugging Face sentence-embedding model name."""
    index_size: int = 4
    """N: skills whose name and when-to-use are put into a prompt."""

    # ---- admission ----
    admit_min_credit: float = 0.0
    """A candidate is admitted when its credit is above this."""
    dedup_threshold: float = 0.9
    """Cosine similarity of (when-to-use + procedure) above which two skills count as the same."""
    kappa: float = 1.0
    """Shrinkage of the credit estimator; used here to pool the re-tests of one step."""

    # ---- re-test ----
    retest_every: int = 2
    """A skill may be re-tested once this many steps passed since it was last verified."""
    retest_per_unit: int = 1
    """Skills re-tested by one rollout group (taken from the group's own task family)."""
    delete_credit: float = 0.02
    """A re-test at or below this counts as "about zero or negative"."""
    delete_strikes: int = 2
    """Delete after this many such re-tests in a row: one noisy measurement must not delete a good skill."""

    # ---- merge ----
    merge_every: int = 4
    """Merges are attempted on steps divisible by this (0 disables merging)."""
    merge_threshold: float = 0.8
    """Similarity above which skills of one family form a merge cluster."""
    merge_max_cluster: int = 3
    merge_cooldown: int = 8
    """Steps before a cluster whose merge was rejected is tried again."""

    # ---- capacity ----
    max_skills_per_family: int = 8


@dataclass
class Skill:
    skill_id: str
    name: str
    when_to_use: str
    procedure: str
    family: str
    """Task type the skill was written on."""
    source_task_id: str
    created_step: int
    source_task_ids: list[str] = field(default_factory=list)
    verified_step: int = -1
    """Step of the last measurement (admission or re-test)."""
    utility: float = 0.0
    """Current utility: the credit of the last measurement."""
    n_loads: int = 0
    """Episodes that loaded it outside of its own tests."""
    loads_since_verified: int = 0
    outcome_sum: float = 0.0
    """Sum over those loads of (episode score - mean score of the episode's group)."""
    n_tests: int = 0
    strikes: int = 0
    last_merge_try: int = -(10**9)
    status: str = "active"  # active | deleted | merged
    removed_reason: str = ""
    merged_from: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def distilled_from(self, task_id: str) -> bool:
        return task_id == self.source_task_id or task_id in self.source_task_ids

    def render(self) -> str:
        """The body the policy reads after ``load_skill``."""
        return f"### {self.name}\nWhen to use: {self.when_to_use}\n{self.procedure}"

    def index_text(self) -> str:
        return f"{self.name}. {self.when_to_use}"

    def content_text(self) -> str:
        return f"{self.when_to_use}\n{self.procedure}"

    def mean_outcome(self) -> float:
        return self.outcome_sum / self.n_loads if self.n_loads else 0.0

    def entry(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "when_to_use": self.when_to_use,
            "text": self.render(),
            "family": self.family,
            "utility": round(self.utility, 4),
        }


class SkillLibrary:
    def __init__(self, config: Optional[SkillLibraryConfig | dict] = None):
        if config is None:
            config = SkillLibraryConfig()
        elif not isinstance(config, SkillLibraryConfig):
            known = set(SkillLibraryConfig.__dataclass_fields__)
            config = SkillLibraryConfig(**{k: v for k, v in dict(config).items() if k in known})
        self.config = config
        self.skills: dict[str, Skill] = {}
        self.version = 0
        self._next_id = 0
        self._embedder = None
        self._vectors: dict[str, list[float]] = {}
        self._staged: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._claimed: set[str] = set()
        self.commit_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        if missing:
            if self._embedder is None:
                self._embedder = make_embedder(self.config.embedder)
            for text, vec in zip(missing, self._embedder.encode(missing), strict=True):
                self._vectors[text] = vec
        return [self._vectors[t] for t in texts]

    def cluster_texts(self, texts: list[str], threshold: Optional[float] = None) -> list[int]:
        """Greedy de-duplication: for each text, the index of the first earlier text it repeats."""
        threshold = self.config.dedup_threshold if threshold is None else threshold
        vectors = self.embed(texts)
        reps: list[int] = []
        out = []
        for i, vec in enumerate(vectors):
            match = next((r for r in reps if cosine(vec, vectors[r]) >= threshold), None)
            if match is None:
                reps.append(i)
                match = i
            out.append(match)
        return out

    # ------------------------------------------------------------------ read

    def _active(self) -> list[Skill]:
        return [s for s in self.skills.values() if s.status == "active"]

    def retrieve(
        self, query: str, task_id: str = "", n: Optional[int] = None, exclude_ids: tuple[str, ...] = ()
    ) -> list[dict[str, Any]]:
        """Coarse recall for one task: the ``n`` most similar skills, ordered by current utility.

        Skills written on this very task instance are left out: a skill must never act as an
        answer cache for the task it came from.
        """
        n = self.config.index_size if n is None else n
        pool = [
            s for s in self._active() if s.skill_id not in exclude_ids and not (task_id and s.distilled_from(task_id))
        ]
        if not pool or n <= 0:
            return []
        vectors = self.embed([query] + [s.index_text() for s in pool])
        sims = {s.skill_id: cosine(vectors[0], v) for s, v in zip(pool, vectors[1:], strict=True)}
        top = sorted(pool, key=lambda s: (-sims[s.skill_id], s.skill_id))[:n]
        top.sort(key=lambda s: (-s.utility, -sims[s.skill_id], s.skill_id))
        return [{**s.entry(), "similarity": round(sims[s.skill_id], 4)} for s in top]

    # ------------------------------------------------------------------ plans

    def retest_plan(self, family: str, step: int, k: Optional[int] = None) -> list[dict[str, Any]]:
        """Skills of ``family`` this group should re-test now (each skill is claimed once per step).

        Priority: verified long ago, loaded often since, and bad outcomes when loaded.
        """
        cfg = self.config
        k = cfg.retest_per_unit if k is None else k
        due = [
            s
            for s in self._active()
            if s.family == family
            and s.skill_id not in self._claimed
            and cfg.retest_every > 0
            and step - s.verified_step >= cfg.retest_every
        ]

        def priority(s: Skill) -> float:
            stale = (step - s.verified_step) / max(cfg.retest_every, 1)
            return stale + math.log1p(s.loads_since_verified) + 5.0 * max(0.0, -s.mean_outcome())

        due.sort(key=lambda s: (-priority(s), s.skill_id))
        chosen = due[: max(k, 0)]
        self._claimed.update(s.skill_id for s in chosen)
        return [s.entry() for s in chosen]

    def merge_plan(self, family: str, step: int) -> Optional[list[dict[str, Any]]]:
        """A cluster of similar skills of ``family`` to try to merge this step, or None."""
        cfg = self.config
        if cfg.merge_every <= 0 or step % cfg.merge_every != 0:
            return None
        pool = [
            s
            for s in self._active()
            if s.family == family and s.skill_id not in self._claimed and step - s.last_merge_try >= cfg.merge_cooldown
        ]
        if len(pool) < 2:
            return None
        vectors = self.embed([s.content_text() for s in pool])
        for i, seed in enumerate(pool):
            cluster = [seed] + [
                other
                for j, other in enumerate(pool)
                if j != i and cosine(vectors[i], vectors[j]) >= cfg.merge_threshold
            ]
            if len(cluster) >= 2:
                cluster = cluster[: cfg.merge_max_cluster]
                self._claimed.update(s.skill_id for s in cluster)
                return [s.entry() for s in cluster]
        return None

    # ------------------------------------------------------------------ stage

    def propose(self, proposal: dict[str, Any]) -> None:
        """Candidates of one group: ``{"family", "source_task_id", "step", "candidates": [{"name",
        "when_to_use", "procedure", "credit", ...}]}``."""
        self._staged["proposals"].append(proposal)

    def report_loads(self, loads: list[dict[str, Any]]) -> None:
        """Ordinary use: ``[{"skill_id", "n", "outcome_sum"}]``."""
        self._staged["loads"].extend(loads)

    def report_retest(self, result: dict[str, Any]) -> None:
        """``{"skill_id", "numerator", "weight", ...}`` of one re-test (see ``credit.Credit``)."""
        self._staged["retests"].append(result)

    def report_merge(self, result: dict[str, Any]) -> None:
        """``{"member_ids", "merged": {...}, "merged_credit", "members_credit", "family"}``."""
        self._staged["merges"].append(result)

    # ------------------------------------------------------------------ commit

    def commit(self, step: int) -> dict[str, Any]:
        cfg = self.config
        stats: dict[str, Any] = Counter()
        events: list[dict[str, Any]] = []

        # 1. ordinary use
        for load in self._staged["loads"]:
            skill = self.skills.get(load["skill_id"])
            if skill is None or skill.status != "active":
                continue
            skill.n_loads += int(load["n"])
            skill.loads_since_verified += int(load["n"])
            skill.outcome_sum += float(load["outcome_sum"])
            stats["loads"] += int(load["n"])

        # 2. re-tests: the same credit estimator as admission, pooled over this step's re-tests
        by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in self._staged["retests"]:
            by_skill[result["skill_id"]].append(result)
        for skill_id, results in by_skill.items():
            skill = self.skills.get(skill_id)
            if skill is None or skill.status != "active":
                continue
            value = pooled_credit([(r["numerator"], r["weight"]) for r in results], cfg.kappa)
            skill.utility, skill.verified_step = value, step
            skill.n_tests += 1
            skill.loads_since_verified = 0
            skill.strikes = skill.strikes + 1 if value <= cfg.delete_credit else 0
            skill.history.append({"step": step, "event": "retest", "credit": round(value, 4)})
            events.append(
                {"event": "retest", "skill_id": skill_id, "credit": round(value, 4), "strikes": skill.strikes}
            )
            stats["retested"] += 1
            if skill.strikes >= cfg.delete_strikes:
                self._remove(skill, "deleted", "retest_not_positive", step, events)
                stats["deleted_retest"] += 1

        # 3. merges: the merged skill replaces its cluster when it is at least as good
        for result in self._staged["merges"]:
            members = [self.skills.get(i) for i in result["member_ids"]]
            if any(m is None or m.status != "active" for m in members):
                continue
            for m in members:
                m.last_merge_try = step
            accepted = result.get("merged") and result["merged_credit"] >= result["members_credit"]
            events.append(
                {
                    "event": "merge" if accepted else "merge_rejected",
                    "members": result["member_ids"],
                    "merged_credit": round(result.get("merged_credit", 0.0), 4),
                    "members_credit": round(result.get("members_credit", 0.0), 4),
                }
            )
            if not accepted:
                stats["merge_rejected"] += 1
                continue
            merged = self._insert(result["merged"], result["family"], members[0].source_task_id, step)
            merged.utility = float(result["merged_credit"])
            merged.merged_from = [m.skill_id for m in members]
            merged.source_task_ids = sorted({t for m in members for t in [m.source_task_id, *m.source_task_ids] if t})
            merged.n_loads = sum(m.n_loads for m in members)
            for m in members:
                self._remove(m, "merged", f"merged_into:{merged.skill_id}", step, events)
            stats["merged"] += 1

        # 4. admission
        for proposal in self._staged["proposals"]:
            for cand in sorted(proposal.get("candidates", []), key=lambda c: -float(c["credit"])):
                stats["proposed"] += 1
                stats[self._admit(cand, proposal, step, events)] += 1

        # 5. capacity
        by_family: dict[str, list[Skill]] = defaultdict(list)
        for s in self._active():
            by_family[s.family].append(s)
        for pool in by_family.values():
            pool.sort(key=lambda s: (s.utility, s.created_step, s.skill_id))
            for s in pool[: max(len(pool) - cfg.max_skills_per_family, 0)]:
                self._remove(s, "deleted", "capacity", step, events)
                stats["deleted_capacity"] += 1

        self._staged.clear()
        self._claimed.clear()
        self.version += 1
        active = self._active()
        summary = {
            "step": step,
            "version": self.version,
            "num_active": len(active),
            "num_total": len(self.skills),
            "mean_utility": sum(s.utility for s in active) / len(active) if active else 0.0,
            "per_family": dict(Counter(s.family for s in active)),
            **{k: int(v) for k, v in stats.items()},
            "events": events,
        }
        self.commit_log.append(summary)
        return summary

    def _admit(self, cand: dict[str, Any], proposal: dict[str, Any], step: int, events: list) -> str:
        cfg = self.config
        credit = float(cand["credit"])
        if not cand.get("name") or not cand.get("procedure"):
            return "rejected_malformed"
        if credit <= cfg.admit_min_credit:
            return "rejected_no_credit"
        # near-duplicate of a library skill: keep only the one with the higher credit
        active = self._active()
        if active:
            text = f"{cand['when_to_use']}\n{cand['procedure']}"
            vectors = self.embed([text] + [s.content_text() for s in active])
            sim, twin = max(
                ((cosine(vectors[0], v), s) for s, v in zip(active, vectors[1:], strict=True)), key=lambda p: p[0]
            )
            if sim >= cfg.dedup_threshold:
                source = proposal.get("source_task_id", "")
                if source and not twin.distilled_from(source):
                    twin.source_task_ids.append(source)
                if credit <= twin.utility:
                    events.append({"event": "duplicate_dropped", "skill_id": twin.skill_id, "sim": round(sim, 3)})
                    return "duplicate_dropped"
                twin.history.append({"step": step, "event": "text_replaced", "old_procedure": twin.procedure})
                twin.name, twin.when_to_use, twin.procedure = cand["name"], cand["when_to_use"], cand["procedure"]
                twin.utility, twin.verified_step, twin.strikes = credit, step, 0
                events.append({"event": "duplicate_replaced", "skill_id": twin.skill_id, "credit": round(credit, 3)})
                return "duplicate_replaced"
        skill = self._insert(cand, proposal.get("family", ""), proposal.get("source_task_id", ""), step)
        skill.utility = credit
        events.append({"event": "admit", "skill_id": skill.skill_id, "name": skill.name, "credit": round(credit, 3)})
        return "admitted"

    def _insert(self, cand: dict[str, Any], family: str, source_task_id: str, step: int) -> Skill:
        skill = Skill(
            skill_id=f"sk{self._next_id:04d}",
            name=cand["name"],
            when_to_use=cand["when_to_use"],
            procedure=cand["procedure"],
            family=family,
            source_task_id=source_task_id,
            created_step=step,
            verified_step=step,
        )
        skill.history.append({"step": step, "event": "created"})
        self._next_id += 1
        self.skills[skill.skill_id] = skill
        return skill

    def _remove(self, skill: Skill, status: str, reason: str, step: int, events: list) -> None:
        skill.status, skill.removed_reason = status, reason
        skill.history.append({"step": step, "event": f"{status}:{reason}"})
        events.append({"event": status, "skill_id": skill.skill_id, "reason": reason})

    # ------------------------------------------------------------------ persistence / inspection

    def stats(self) -> dict[str, Any]:
        active = self._active()
        return {
            "version": self.version,
            "num_active": len(active),
            "num_total": len(self.skills),
            "mean_utility": sum(s.utility for s in active) / len(active) if active else 0.0,
            "per_family": dict(Counter(s.family for s in active)),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "next_id": self._next_id,
            "config": asdict(self.config),
            "skills": [asdict(s) for s in self.skills.values()],
            "commit_log": self.commit_log,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.version = state["version"]
        self._next_id = state["next_id"]
        self.skills = {s["skill_id"]: Skill(**s) for s in state["skills"]}
        self.commit_log = state.get("commit_log", [])
        self._staged.clear()
        self._claimed.clear()

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.state_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return path

    def load(self, path: str) -> None:
        with open(path) as f:
            self.load_state_dict(json.load(f))
