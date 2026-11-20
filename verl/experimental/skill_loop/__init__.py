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
"""Skill-based self-evolution on top of the V1 agent loop.

Heavy submodules (TransferQueue worker, trainer) are imported by their fully qualified names from
the Hydra config; importing this package only exposes the dependency-free pieces.
"""

from .skill_advantage import compute_skill_advantage
from .skill_library import Skill, SkillLibrary, SkillLibraryConfig

__all__ = ["Skill", "SkillLibrary", "SkillLibraryConfig", "compute_skill_advantage"]
