#!/usr/bin/env bash
# GRPO with skill-based self-evolution on ALFWorld (text) | FSDP actor + vLLM rollout, colocated.
#
#   bash examples/skill_evolve/run_alfworld_skill_grpo.sh                 # skill evolution
#   SKILL=0 bash examples/skill_evolve/run_alfworld_skill_grpo.sh         # plain GRPO baseline
#   ROLLOUT_ONLY=1 bash examples/skill_evolve/run_alfworld_skill_grpo.sh  # validation rollouts only
#
# Run from the verl repo root. Prerequisites: see docs/skill_evolve/README.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
source examples/skill_evolve/common.sh

NGPUS=${NGPUS:-1}
TOTAL_STEPS=${TOTAL_STEPS:-60}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-20}        # the last step is always saved; -1 disables saving
ACTOR_LR=${ACTOR_LR:-1e-6}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-16384}

NUM_CHECKS=${NUM_CHECKS:-2}        # m: check tasks per unit
CONTROL_N=${CONTROL_N:-4}          # M: control episodes per check task
CHECK_N=${CHECK_N:-4}              # K/m: episodes per new skill per check task
MAX_CANDIDATES=${MAX_CANDIDATES:-4}
if [ "${SKILL:-1}" = 0 ]; then
    # stock worker + stock trainer: the same agent loop, one episode per session, no library
    SKILL_EVOLVE=(actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/skill_evolve/config/skill_agent_baseline.yaml)
    ROWS_PER_GROUP=${ROLLOUT_N}
else
    SKILL_EVOLVE+=(skill_evolve.num_checks=${NUM_CHECKS} skill_evolve.control_n=${CONTROL_N}
                   skill_evolve.check_n=${CHECK_N} skill_evolve.max_candidates=${MAX_CANDIDATES})
    # upper bound of a unit's episodes: G on the source task, and per check task M control episodes plus
    # K/m episodes for each new skill (re-test and merge arms add a few more on some steps)
    ROWS_PER_GROUP=$((ROLLOUT_N + NUM_CHECKS * (CONTROL_N + MAX_CANDIDATES * CHECK_N)))
fi
# verl sizes an optimizer mini-batch as ppo_mini_batch_size * rollout.n ROWS. Cover the whole step
# with one mini-batch so the skill run and the baseline take the same number of optimizer steps
# (one per trainer step); otherwise the skill run, with several times the rows, would take several times the updates.
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$(( (GROUPS_PER_STEP * ROWS_PER_GROUP + ROLLOUT_N - 1) / ROLLOUT_N ))}

MODEL=(
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    # the actor shares its GPU with the inference engine: keep it off the GPU while rolling out
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ENGINE=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_model_len=8448
    actor_rollout_ref.rollout.max_num_seqs=256
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN}
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
)

RUN=(
    trainer.n_gpus_per_node=${NGPUS}
    trainer.nnodes=1
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.total_training_steps=${TOTAL_STEPS}
    trainer.total_epochs=1000
    trainer.test_freq=${TEST_FREQ}
    trainer.save_freq=${SAVE_FREQ}
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}
    trainer.validation_data_dir="${OUT_DIR}/validation"
)
if [ "${ROLLOUT_ONLY:-0}" = 1 ]; then
    RUN+=(trainer.val_only=True trainer.val_before_train=True)
fi

python3 -m verl.trainer.main_ppo \
    "${SKILL_EVOLVE[@]}" "${DATA[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}" \
    "${MODEL[@]}" "${ACTOR[@]}" "${ENGINE[@]}" "${RUN[@]}" \
    "$@"
echo "trajectories: ${OUT_DIR}/trajectories   library: ${OUT_DIR}/checkpoints/skill_library"
