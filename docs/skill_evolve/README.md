# Skill-based self-evolution on verl

这个 fork 在 verl（`a4e0dd2`，V1 trainer）上加了一条支持 **"生成 skill → 沉淀进 skill library → 后续 rollout 加载 skill"** 闭环的 rollout。一个 prompt 不再是 n 条互相独立的轨迹，而是一个训练单元 $(A;\ B_1, \dots, B_m)$：模型在 A 上做题并自己决定写 skill，新 skill 当步就在同题型的检验题 B 上测出功劳，功劳是 `write_skill` 调用的 reward。

| 文档 | 内容 |
|---|---|
| [`DESIGN.md`](DESIGN.md) | benchmark 与训练集、reward 设计与归因（核心）、skill 机制、harness 改造（diff 级） |
| [`VALIDATION.md`](VALIDATION.md) | 跑了什么、结果是什么、带 reward 的完整 rollout 轨迹在哪 |
| [`REFLECTION.md`](REFLECTION.md) | 主要缺点、scale 瓶颈、改进方向，以及对这道题本身的反思 |

## 代码在哪

```
verl/experimental/skill_loop/         新增的包（功劳、库、agent loop、单元级编排 worker、advantage、trainer）
verl/trainer/config/skill_evolve/     Hydra 配置组（每个参数都有注释）
examples/data_preprocess/alfworld_skill_chains.py   训练集构造
examples/skill_evolve/                启动脚本、轨迹报告渲染、功劳估计的检验、脚本策略、vLLM HTTP 客户端
tests/experimental/skill_loop/        52 个 CPU 测试
```

对 verl 已有代码的修改只有三处扩展点（+45/−14 行）：`verl/trainer/main_ppo.py`、`verl/trainer/ppo/v1/agent_loop_tq.py`、`verl/trainer/ppo/v1/trainer_base.py`。用 `git diff a4e0dd2 -- verl/trainer` 查看。

## 0. 环境（一次性）

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[vllm]" && uv pip install -r requirements.txt
uv pip install alfworld

export ALFWORLD_DATA=/path/to/alfworld_data
alfworld-download                         # 约 1.5 GB，只用到文本游戏

python examples/data_preprocess/alfworld_skill_chains.py --local_save_dir ~/data/alfworld_units
```

之后所有脚本读这几个环境变量（见 `examples/skill_evolve/common.sh`）：

```bash
export ALFWORLD_DATA=...                  # alfworld-download 的目录
export DATA_DIR=~/data/alfworld_units     # 上一步的输出
export MODEL_PATH=Qwen/Qwen3.5-4B         # 或本地目录
export OUT_DIR=./outputs/skill_evolve/run1
```

## 1. 起 rollout

下面几种方式跑的是同一套 rollout 代码（dataloader → TransferQueue → Ray agent loop worker → ALFWorld → ReplayBuffer → advantage → 库提交），区别只在 token 由谁生成。都在 verl 仓库根目录执行。

**(a) 真实模型，只 rollout 不训练**（需要 1 张 GPU 跑 vLLM）：

```bash
vllm serve $MODEL_PATH --served-model-name policy --port 8000 --max-model-len 8448 &
GROUPS_PER_STEP=4 STEPS=4 VALIDATE=false bash examples/skill_evolve/run_rollout_scripted_cpu.sh \
    +driver.llm=vllm_http +driver.base_url=http://127.0.0.1:8000/v1 +driver.model=policy
```

**(b) 没有 GPU，脚本策略**（不是模型；用来验证 harness）：

```bash
STEPS=6 bash examples/skill_evolve/run_rollout_scripted_cpu.sh \
    skill_evolve.library.embedder=hash skill_evolve.library.merge_threshold=0.6 skill_evolve.library.merge_every=2
```

**(c) GRPO 训练**（verl 自己的 vLLM 引擎 + FSDP actor，同卡）。启动脚本已按新配置更新，但这条路径没有运行过：

```bash
TOTAL_STEPS=60 bash examples/skill_evolve/run_alfworld_skill_grpo.sh
SKILL=0 bash examples/skill_evolve/run_alfworld_skill_grpo.sh      # 不带 skill 的 GRPO baseline（原版 worker + 原版 trainer）
```

常用旋钮（环境变量）：`GROUPS_PER_STEP`（每步多少个训练单元，默认 8）、`ROLLOUT_N`（A 的轨迹数 G，默认 8）、`AGENT_WORKERS`（环境并行度）。其余参数直接追加 Hydra override，例如 `skill_evolve.check_n=8 skill_evolve.tau=0.05`，全部参数见 `verl/trainer/config/skill_evolve/default.yaml`。

## 2. 看结果

每次运行在 `$OUT_DIR` 下产生：

```
trajectories/groups_worker*.jsonl     每个训练单元一条记录：各组的索引、完整对话、加载和写作、使用权重、功劳、reward
trajectories/advantages.jsonl         trainer 给每条轨迹算的 advantage，以及 write_skill token 上的 advantage
checkpoints/skill_library/step_N.json 每步提交后的库（skill 文本、效用、使用次数、复测次数、事件历史）
summary.json                          每步指标（仅 rollout-only 驱动）
```

渲染成可读报告（挑出"写出并检验新 skill / 加载库里的 skill 并复测 / 没拿到功劳的 skill / 合并 / 验证"几类单元）：

```bash
python examples/skill_evolve/render_trajectories.py --run_dir $OUT_DIR --out $OUT_DIR/ROLLOUT_TRAJECTORIES.md
```

值得盯的指标：

| 指标 | 含义 |
|---|---|
| `skill/write_frac` | A 的轨迹里写了 skill 的比例 |
| `skill/write_credit_mean`、`skill/write_credit_pos_frac` | 新 skill 的平均功劳，以及功劳为正的比例 |
| `skill/source_load_frac`、`skill/control_load_frac` | 做题时加载了库里 skill 的轨迹比例 |
| `skill/candidate_loaded_frac` | 检验题上，实验组轨迹加载了被测新 skill 的比例 |
| `skill/candidate_influence`、`skill/candidate_usage_weight` | 加载后的平均影响力和使用权重 |
| `skill/usage_weight_corr_steps`、`skill/usage_weight_corr_score` | 使用权重和轨迹长度、得分的相关性（诊断用，见 REFLECTION） |
| `skill/candidate_lift` | 实验组平均分减对照组平均分（不加权） |
| `skill_lib/num_active`、`admitted`、`rejected_no_credit`、`duplicate_*`、`retested`、`deleted_*`、`merged` | 库的生命周期 |

## 3. 测试

```bash
pytest tests/experimental/skill_loop -q        # 52 个，约 10 秒，不需要 GPU、模型或 ALFWorld
```

`test_skill_loop_group_on_cpu.py` 是端到端的：用一个"每种颜色的门有一个秘密按钮"的玩具环境和脚本策略，断言新 skill 的功劳等于公式手算的值、`write_skill` 的 token 上带着归一化后的功劳、没被加载的 skill 功劳为 0、加载后被无视的 skill 功劳为 0、把人带偏的 skill 功劳为负、重复写库里已有的 skill 功劳约为 0、库里的坏 skill 连续两次复测不为正后被删除。

## 4. 换一个 benchmark

实现一个模块，暴露 `make_env(task: TaskSpec, **kwargs) -> TextEnv` 和字符串 `ACTION_GUIDE`（见 `envs/base.py`、`envs/alfworld_env.py`），数据集每行给出 `extra_info.source`（A）和 `extra_info.probes`（同题型的检验题），然后 `skill_evolve.agent.env_module=<你的模块>`。库、编排、reward、trainer 都不用动。
