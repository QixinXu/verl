#!/usr/bin/env bash
# Skill-evolution rollout on a CPU node, driven by a SCRIPTED policy instead of a model.
#
# Runs the real data path (dataloader -> TransferQueue -> Ray agent loop workers -> ALFWorld ->
# replay buffer -> role-grouped advantages -> library commit) for a few steps and writes the
# trajectories with their rewards and advantages. No GPU, no gradient. Run from the verl repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."
source examples/skill_evolve/common.sh

STEPS=${STEPS:-4}
python3 examples/skill_evolve/rollout_only.py \
    "${SKILL_EVOLVE[@]}" "${DATA[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}" \
    actor_rollout_ref.rollout.name=vllm \
    algorithm.rollout_correction.bypass_mode=True \
    trainer.logger=console \
    +driver.steps=${STEPS} +driver.validate=${VALIDATE:-true} +driver.out="${OUT_DIR}" \
    "$@"
echo "trajectories: ${OUT_DIR}/trajectories   library: ${OUT_DIR}/checkpoints/skill_library"
