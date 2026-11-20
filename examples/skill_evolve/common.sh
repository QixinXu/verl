# Shared settings of the skill-evolution examples. Source it; do not execute it.
#
# Paths (override through the environment):
#   MODEL_PATH       HF model directory or hub id of the policy
#   DATA_DIR         output of examples/data_preprocess/alfworld_skill_chains.py
#   ALFWORLD_DATA    output of `alfworld-download`
#   OUT_DIR          checkpoints, library snapshots and trajectory dumps of this run

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B}
DATA_DIR=${DATA_DIR:-$HOME/data/alfworld_skill}
OUT_DIR=${OUT_DIR:-$PWD/outputs/skill_evolve/$(date +%Y%m%d_%H%M%S)}
export ALFWORLD_DATA=${ALFWORLD_DATA:?set ALFWORLD_DATA to the alfworld-download directory}

GROUPS_PER_STEP=${GROUPS_PER_STEP:-8}   # prompts (task chains) per training step
ROLLOUT_N=${ROLLOUT_N:-8}               # sessions per chain = GRPO group size
AGENT_WORKERS=${AGENT_WORKERS:-8}       # agent loop worker processes (one env thread each)
VAL_FILE=${VAL_FILE:-$DATA_DIR/val_mixed_small.parquet}

# The three switches that turn stock verl into the skill-evolution harness.
SKILL_EVOLVE=(
    +skill_evolve=default
    +trainer.v1.trainer_class=verl.experimental.skill_loop.skill_trainer.SkillEvolveTrainerSync
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.skill_loop.skill_loop_tq.SkillLoopManagerTQ
    skill_evolve.dump_dir="${OUT_DIR}/trajectories"
)

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${DATA_DIR}/train.parquet"
    data.val_files="${VAL_FILE}"
    # the parquet is written in curriculum order; shuffling would destroy it
    data.shuffle=False
    data.train_batch_size=${GROUPS_PER_STEP}
    data.dataloader_num_workers=0
    # a trajectory is a whole episode: system prompt + up to 30 (thought, action, observation) turns
    data.max_prompt_length=2048
    data.max_response_length=6144
    +data.apply_chat_template_kwargs.enable_thinking=False
)

ROLLOUT=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.agent.default_agent_loop=skill_agent
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_WORKERS}
    actor_rollout_ref.rollout.calculate_log_probs=True
    reward.num_workers=1
)

TRAINER=(
    trainer.default_local_dir="${OUT_DIR}/checkpoints"
    trainer.project_name=skill_evolve
    trainer.experiment_name=alfworld
)
