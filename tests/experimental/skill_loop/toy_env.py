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
"""A tiny hidden-rule environment for testing the skill loop without a model or a simulator.

Every task family has a secret button. The door only opens after that button was pressed, and the
secret is the same for all task instances of the family: exactly the kind of knowledge a skill can
carry from one task to another.
"""

from verl.experimental.skill_loop.envs.base import StepResult, TaskSpec, TextEnv

ACTION_GUIDE = "Commands: press <a|b|c>, open door"
SECRET = {"red": "a", "blue": "b", "green": "c"}


class ToyDoorEnv(TextEnv):
    def __init__(self, task: TaskSpec, max_steps: int = 6):
        self.task = task
        self.max_steps = max_steps
        self.pressed = None

    def reset(self) -> str:
        self.pressed = None
        colour = self.task.family
        return f"You face a {colour} door with buttons a, b, c.\nYour task is to: open the {colour} door"

    def step(self, command: str) -> StepResult:
        if command.startswith("press ") and command[6:] in ("a", "b", "c"):
            self.pressed = command[6:]
            return StepResult(f"You press {self.pressed}.", done=False, won=False)
        if command == "open door":
            if self.pressed == SECRET[self.task.family]:
                return StepResult("The door opens.", done=True, won=True)
            return StepResult("The door stays shut.", done=False, won=False)
        return StepResult("Nothing happens.", done=False, won=False, valid=False)


def make_env(task: TaskSpec, **kwargs) -> ToyDoorEnv:
    return ToyDoorEnv(task, **kwargs)
