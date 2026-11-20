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
"""Drive the skill-evolution rollout without a model engine (see ``skill_loop/rollout_only.py``).

Takes the same Hydra overrides as ``verl.trainer.main_ppo`` plus::

    +driver.steps=4           number of rollout steps (no gradient is taken)
    +driver.validate=true     run validation before the first and after the last step
    +driver.seed=0            seed of the scripted policy
    +driver.out=<dir>         where to write summary.json
    +driver.llm=scripted      scripted ALFWorld policy (``scripted_alfworld_policy.py``) -- NOT a model
    +driver.llm=vllm_http     the real model behind an external ``vllm serve``;
                              needs +driver.base_url=http://host:port/v1 and +driver.model=<served name>
"""

import json
import os
import sys

import hydra
import ray
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    import transfer_queue as tq
    from scripted_alfworld_policy import ScriptedAlfworldLLM

    from verl.experimental.skill_loop.rollout_only import SkillRolloutOnlyTrainer
    from verl.experimental.skill_loop.skill_loop_tq import SkillLoopManagerTQ
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    driver = OmegaConf.to_container(config.get("driver", {}), resolve=True)
    steps, seed = int(driver.get("steps", 3)), int(driver.get("seed", 0))
    out_dir = driver.get("out") or config.trainer.default_local_dir

    runtime_env = OmegaConf.to_container(OmegaConf.create(get_ppo_ray_runtime_env(config)), resolve=True)
    runtime_env.setdefault("env_vars", {})["TRANSFER_QUEUE_ENABLE"] = "1"
    runtime_env["env_vars"]["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(os.path.abspath(__file__)), os.environ.get("PYTHONPATH", "")]
    )
    ray.init(runtime_env=runtime_env, num_cpus=int(driver.get("num_cpus", os.cpu_count())))

    config.transfer_queue.enable = True
    OmegaConf.resolve(config)
    tq.init(config.transfer_queue)
    try:
        trainer = SkillRolloutOnlyTrainer(config=config)
        trainer.init()
        if driver.get("llm", "scripted") == "vllm_http":
            # the real model, served by an external `vllm serve` process
            from vllm_http_client import VLLMHttpClient

            llm = VLLMHttpClient(
                base_url=driver["base_url"],
                model=driver.get("model", config.actor_rollout_ref.model.path),
                model_path=config.actor_rollout_ref.model.path,
                max_concurrency=int(driver.get("max_concurrency", 64)),
            )
        else:
            llm = ScriptedAlfworldLLM(model_path=config.actor_rollout_ref.model.path, seed=seed)
        trainer.agent_loop_manager = SkillLoopManagerTQ.create(
            config=config, llm_client=llm, teacher_client=None, reward_loop_worker_handles=None
        )

        summary = {"steps": []}
        if driver.get("validate", False):
            summary["val_before"] = trainer.validate()
            print("[rollout_only] validation before:", json.dumps(summary["val_before"], indent=1))
        for _ in range(steps):
            metrics = trainer.rollout_step()
            library = metrics.pop("library")
            line = {k: round(float(v), 4) for k, v in metrics.items() if k.startswith("skill/") or k == "rows"}
            line["library"] = {k: v for k, v in library.items() if k != "events"}
            line["library_events"] = library["events"]
            summary["steps"].append({"step": trainer.global_steps, **line})
            print(f"[rollout_only] step {trainer.global_steps}:", json.dumps(line))
        if driver.get("validate", False):
            summary["val_after"] = trainer.validate()
            print("[rollout_only] validation after:", json.dumps(summary["val_after"], indent=1))

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        trainer.close_dataloaders()
    finally:
        tq.close()


if __name__ == "__main__":
    main()
