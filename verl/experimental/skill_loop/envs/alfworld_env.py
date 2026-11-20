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
"""ALFWorld (text-only) adapter for the skill loop.

Only the TextWorld engine is used: no THOR, no display, no detector weights. One game file is one
task instance; success is the simulator's goal condition, so the reward needs no LLM judge.
"""

from __future__ import annotations

import os

from verl.experimental.skill_loop.envs.base import StepResult, TaskSpec, TextEnv

# The six ALFWorld task types, in the short names used by the paper.
TASK_TYPES = {
    "pick_and_place_simple": "pick",
    "look_at_obj_in_light": "look",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_two_obj_and_place": "pick2",
}

ACTION_GUIDE = """Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it."""  # noqa: E501

_BANNER = "-= Welcome to TextWorld, ALFRED! =-"


def data_root() -> str:
    root = os.environ.get("ALFWORLD_DATA")
    if not root:
        raise RuntimeError("ALFWORLD_DATA is not set; run `alfworld-download` and export it.")
    return os.path.join(root, "json_2.1.1")


class AlfworldTextEnv(TextEnv):
    """One ALFWorld game. ``payload["gamefile"]`` is relative to ``$ALFWORLD_DATA/json_2.1.1``."""

    def __init__(self, task: TaskSpec, max_steps: int = 30, admissible_hint: str = "on_invalid", max_hint: int = 12):
        assert admissible_hint in ("never", "on_invalid", "always")
        self.task = task
        self.max_steps = max_steps
        self.admissible_hint = admissible_hint
        self.max_hint = max_hint
        self._env = None
        self._admissible: list[str] = []

    def reset(self) -> str:
        # Imported lazily: textworld/alfworld are only needed inside agent loop workers.
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
        from textworld.gym.envs import TextworldGymEnv

        gamefile = self.task.payload["gamefile"]
        if not os.path.isabs(gamefile):
            gamefile = os.path.join(data_root(), gamefile)
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True)
        self._env = TextworldGymEnv(
            gamefiles=[gamefile],
            request_infos=request_infos,
            # the step budget is enforced here, not by textworld, so that `done` always means `won`
            max_episode_steps=self.max_steps + 5,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        )
        obs, infos = self._env.reset()
        self._admissible = list(infos.get("admissible_commands") or [])
        obs = obs.replace(_BANNER, "").strip()
        if self.admissible_hint == "always":
            obs = f"{obs}\n{self._hint()}"
        return obs

    def step(self, command: str) -> StepResult:
        obs, _score, done, infos = self._env.step(command)
        won = bool(infos.get("won", False))
        self._admissible = list(infos.get("admissible_commands") or [])
        obs = obs.strip()
        valid = obs != "Nothing happens."
        if (not valid and self.admissible_hint == "on_invalid") or self.admissible_hint == "always":
            obs = f"{obs}\n{self._hint()}"
        return StepResult(observation=obs, done=bool(done) or won, won=won, valid=valid)

    def _hint(self) -> str:
        # `help` is noise; movement is listed last because it is always available and is the bulk
        # of the list, so the interaction commands (take/open/move/...) survive truncation.
        cmds = [c for c in self._admissible if c != "help"]
        cmds.sort(key=lambda c: (c.startswith("go to "), c))
        shown = cmds[: self.max_hint]
        more = f", ... (+{len(cmds) - len(shown)} more)" if len(cmds) > len(shown) else ""
        return "Commands available here: " + ", ".join(shown) + more

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def make_env(task: TaskSpec, **kwargs) -> AlfworldTextEnv:
    return AlfworldTextEnv(task, **kwargs)
