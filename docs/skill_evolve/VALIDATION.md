# 验证

## 结论

- 内容：单元测试；脚本策略的端到端 rollout（8 步），并在这次 rollout 上用已知答案检验功劳估计；真模型（Qwen3.5-4B，未微调）的 rollout（4 步）。按题目要求只做 rollout，不做梯度更新。
- 结论：harness 端到端跑通；功劳能把有用的 skill 和空话分开，重复的 skill 会被高估（REFLECTION 缺点 1）。真模型零样本就会写 skill，写得能迁移的 skill 拿到功劳并入库，写得太具体的没人加载、功劳为 0。

## 1. 单元测试

```bash
pytest tests/experimental/skill_loop -q     # 52 passed
pytest tests/trainer/ppo/v1 -q              # 264 passed：上游 V1 trainer 的测试，确认三处补丁不改变默认行为
```

端到端测试（`test_skill_loop_group_on_cpu.py`）用一个"每种颜色的门有一个秘密按钮"的玩具环境和脚本策略，不需要模型、ALFWorld 或 Ray。断言的内容：

| 情况 | 断言 |
|---|---|
| 空库，A 上写出 skill，检验题上加载并照着做 | 功劳等于按公式手算的值且为正；只有 `write_skill` 的 token 带功劳项；提交后入库 |
| 检验题上没人加载这条 skill | 功劳为 0，写作 token 的功劳项为 0，不入库 |
| 加载了但动作不依赖它 | 影响力为 0，使用权重为 0，功劳为 0 |
| 加载后被带偏 | 功劳为负，写作 token 的功劳项为负 |
| 重复写库里已有的 skill | 功劳约为 0，不入库；库里那条的使用次数被记上 |
| 库里一条好 skill 和一条坏 skill 被复测 | 好的效用为正；坏的连续两次复测不为正后被删除 |
| 一簇相似 skill | 起草合并版，三个组各自跑，合并版不差才替换 |
| 验证 | 只读库，写出的 skill 丢弃，不暂存 |

## 2. 脚本策略的端到端 rollout

产物在 [`artifacts/scripted_cpu/`](artifacts/scripted_cpu/)：轨迹报告 `ROLLOUT_TRAJECTORIES.md`、每条轨迹的 advantage、每步的库快照、每步指标。

**设置。** 用一段手写的 ALFWorld 策略（`scripted_alfworld_policy.py`）代替模型生成 token 和对数概率，其余都是真实组件：真实 tokenizer 和 chat template、真实 ALFWorld 游戏、Ray worker、TransferQueue、ReplayBuffer、advantage 计算、库的提交。脚本策略被故意写成在几件事上"不会"（先试 `put` 再试 `move`、先去炉子再去微波炉），加载的 skill 正文提到正确做法时才直接做对。它写出的 skill 有正确的、空话的、格式错的。

**规模。** 8 步，每步 8 个训练单元，G = 8，m = 2，M = 4，每个候选每道检验题 4 条。共 64 个训练单元、2184 条训练轨迹，检验了 129 条新 skill。

**每步的情况：**

| 步 | 轨迹数 | 写了 skill 的 A 轨迹 | 新 skill 平均功劳 | 实验组加载被测 skill 的比例 | 库里 skill 数 | 库事件 |
|---|---|---|---|---|---|---|
| 1 | 240 | 67% | +0.064 | 91% | 3 | 入库 3 |
| 2 | 288 | 67% | +0.028 | 24% | 3 | 合并被拒 1 |
| 3 | 264 | 58% | +0.006 | 23% | 3 | 复测 2 |
| 4 | 288 | 69% | +0.052 | 29% | 3 | 复测 1 |
| 5 | 272 | 58% | +0.014 | 32% | 7 | 入库 4（新出现 heat、clean 两个题型） |
| 6 | 272 | 70% | +0.031 | 20% | 7 | 复测 3 |
| 7 | 256 | 44% | +0.004 | 13% | 7 | 复测 4 |
| 8 | 304 | 72% | +0.022 | 21% | 7 | 复测 3 |

第 1 步库是空的，新 skill 是索引里唯一的一条，所以加载比例高、功劳高。之后库里有了等价的 skill，新写的大多是重复，加载比例和功劳都降下来。第 5 步出现新题型，又有新 skill 入库。

**一个单元里 reward 是怎么来的**（报告里的单元 1，第 1 步，题型 look，库为空）：

- A：`look at book under the desklamp`，8 条轨迹，其中 6 条在任务完成后写了 skill，去重后是 2 个候选。
- B1 基线分 0.846，B2 基线分 0.675（各 4 条对照轨迹的平均）。
- 候选 new0 的 8 条实验轨迹全部加载了它。其中 2 条的影响力为负（加载后的动作在没有正文时反而更可能），使用权重为 0；其余 6 条权重为 1。功劳 = (0.054 + 0.037 + 0.225 + 0.225 + 0.242 + 0.242) / (6 + 1) = +0.146。
- 写作 token 的 advantage：三条写了 new0 的轨迹功劳相同，但 advantage 分别是 +0.94、−1.77、+0.94，差别来自它们各自在 A 上的得分。

报告里还有四个单元：加载库里的 skill 并复测一条库里的 skill、没拿到功劳的 skill、一次合并（合并版功劳 +0.030，原簇 +0.036，被拒）、验证。每个单元都附了完整对话。

## 3. 功劳估计靠不靠谱

脚本策略里每条 skill 的真实价值是已知的，可以当标准答案。`analyze_credit.py` 把 129 条被检验的新 skill 分成三类（结果在 `artifacts/scripted_cpu/credit_analysis.json`）：

| 类别 | 条数 | 应该是 | 平均功劳 | 功劳为正 | 不加权的组间分差 |
|---|---|---|---|---|---|
| 有用：正确的做法，库里还没有这个题型的 skill | 24 | 明显为正 | +0.050 | 83% | +0.032 |
| 重复：正确的做法，但库里已有等价的 | 59 | 约 0 | +0.031 | 76% | +0.010 |
| 空话 | 46 | 约 0 | 0.000 | 0% | +0.003 |

- **空话被准确地判成 0。** 加载之后动作没有变化，影响力为 0，使用权重为 0。
- **区分度比不加权好。** 把"有用"和其余两类分开的 AUC：功劳 0.79，只按是否加载加权 0.72，不加权的组间分差 0.67。
- **重复的 skill 被高估。** 平均功劳是有用 skill 的六成，76% 为正。原因和修法见 REFLECTION 缺点 1：基线换成"对照组里真正用了库里 skill 的轨迹"后，重复 skill 的功劳降到 +0.003，有用 skill 是 +0.042。
- **写作 reward 能排对顺序。** 同组里写了有用 skill 的轨迹，写作 token 的 advantage 高于写了重复或空话的轨迹，比例是 78%。

## 4. 真模型的 rollout

产物在 [`artifacts/real_model_rollout/`](artifacts/real_model_rollout/)。模型是 Qwen3.5-4B，没有微调，用 vLLM 在一张 GPU 上推理，不做梯度更新。4 步，每步 4 个训练单元，G = 8，m = 2，M = 4，每个候选每道检验题 4 条，共 664 条轨迹。前两步的题型是 pick，后两步是 look。

| 步 | 题型 | A 轨迹里写了 skill 的比例 | 实验组加载被测 skill 的比例 | 新 skill 功劳为正的比例 | 入库 / 提交 | 做 A 时加载库里 skill 的比例 | 复测 |
|---|---|---|---|---|---|---|---|
| 1 | pick | 91% | 7% | 12.5% | 2 / 16 | 库为空 | — |
| 2 | look | 41% | 16% | 43% | 4 / 10 | 3% | — |
| 3 | pick | 78% | 1% | 4% | 1 / 13 | 16% | 2 条，功劳都是 0 |
| 4 | look | 25% | 6% | 11% | 1 / 6 | 50% | 4 条，功劳 −0.19、−0.14、−0.04、0 |

观察到的事：

- **模型零样本就会调用 `write_skill`，格式正确。** 不做 SFT 也有写作行为可以奖励。
- **写出的 skill 大多太具体，换一道题就没人加载。** 例如 A 是"把手机放到边桌"时，写出的 skill 叫"Find Cellphone""move cellphone to sidetable"，在"把钥匙放到茶几"的检验题上 8 条轨迹没有一条加载，功劳是 0。同一个单元里"Find object in drawer"（适用场景写的是"在几个抽屉里找一个物体并移到别处"）被加载了 1 次，功劳 +0.094，入库。这是设计想要的方向：只有适用场景写得能迁移的 skill 才拿得到功劳。
- **复测在起作用。** 前两步入库的 6 条 skill 在第 3、4 步被复测，功劳全部不为正，各记了一次。按规则再有一次就会被删除。
- **库里的 skill 开始被加载。** 做 A 时加载库里 skill 的轨迹比例从 3% 升到 50%，随库里可选的 skill 变多而上升。

报告里有三个带完整对话的单元：空库时写 skill 并检验、加载库里的 skill 并复测、没拿到功劳的 skill。这次运行里没有触发合并。


## 5. 复现

```bash
# 脚本策略的 rollout（CPU）
STEPS=8 bash examples/skill_evolve/run_rollout_scripted_cpu.sh \
    skill_evolve.library.embedder=hash skill_evolve.library.merge_threshold=0.6 skill_evolve.library.merge_every=2
python examples/skill_evolve/render_trajectories.py --run_dir $OUT_DIR --out report.md
python examples/skill_evolve/analyze_credit.py --run_dir $OUT_DIR
```

本集群上对应 `slurm/scripted_cpu.sbatch`、`slurm/real_gpu.sbatch`。
