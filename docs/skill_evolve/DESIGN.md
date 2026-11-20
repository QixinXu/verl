# Skill-based Self-Evolving Agent：设计文档

基线：`volcengine/verl@a4e0dd2`（V1 trainer + TransferQueue + agent loop）。代码在 `verl/experimental/skill_loop/`。

本文按依赖顺序写：§1 定义所有基本概念，§2 讲 benchmark 和数据，§3 讲 reward，§4 讲 skill 库的维护，§5 把前面的内容串成一个训练步，§6 讲对 verl 的改造。

## 0. 一页概览

目标是让模型在 RL 训练中积累经验：模型做题时可以把学到的做法写成一段文字（skill），这些 skill 存进一个全局的库；之后做同类题时，库里相关的 skill 会列在 prompt 里，模型可以选择读取它们。

要解决的核心问题是怎么给"写 skill"这个行为打 reward。一条 skill 写得好不好，写它的那道题看不出来，要等别的题用了它才知道。本设计的做法是：每写出一条新 skill，在同一个训练步里马上拿几道同类型的其他题检验它，比较"带上这条 skill 做题"和"不带它做题"的得分，只统计模型确实用了这条 skill 的那些轨迹。这个得分差叫这条 skill 的**功劳**（§3.3），作为写 skill 那段 token 的 reward。

当步检验只能反映 skill 刚写出来时的价值。模型在训练中会变，一条 skill 可能过时，或者内容已经被模型学进参数，所以库里的 skill 还会定期用同一个公式重新测量，测出来没有价值的会被删掉（§4）。

---

## 1. 基本概念

### 1.1 题、题型、轨迹、任务得分

- **题**：环境里的一个任务实例，例如 ALFWorld 里"把一支蜡烛放进抽屉"。
- **题型**：同一题型的题解法流程相同，具体物体、位置、房间不同。ALFWorld 有 6 个题型（§2.1）。代码和数据里题型叫 `family`。
- **轨迹**：模型做一道题的一次完整多轮交互。每一轮，模型收到一条观测（环境的文字描述），回复一行思考和一条命令，格式固定为：

  ```
  Thought: <一句话>
  Action: <一条命令>
  ```

  命令执行后环境返回下一条观测。一局在任务完成或步数用完时结束。
- **任务得分**：一条轨迹的得分。成功时为 `1 − 0.5 × 步数 / 步数上限`，失败为 0。步数越少得分越高，所以一条有用的 skill 即使不改变成败，也能通过省步数体现价值。

### 1.2 Skill 和 skill 库

一条 **skill** 是一段文本，三个字段：

```
name:        名字
when_to_use: 什么情况下该用（一句话）
procedure:   具体怎么做（自然语言步骤，最多 120 个词）
```

**skill 库**是所有轨迹共享的一个全局 skill 集合，跨训练步保留。库的内容只在每个训练步结束时更新一次（叫**提交**，§4），一步之内所有轨迹读到的都是同一个版本。下文把"本步开始时的库"叫**旧库**。

### 1.3 索引与检索

模型不会看到整个库。每道题开始时，从库里**检索**出与这道题相关的 N 条 skill（默认 N = 4），把它们的 `name` 和 `when_to_use` 列在 prompt 里，每条前面有一个编号 `S1`、`S2`……。这个列表叫这道题的**索引**。`procedure`（正文）不在索引里，只有模型主动读取时才出现（§1.4）。

检索方法：把每条 skill 的 `name + when_to_use` 和题目描述分别编码成向量，取余弦相似度最高的 N 条。编码器默认 `BAAI/bge-small-en-v1.5`，检索器不参与训练。一条 skill 永远不会出现在写出它的那道题的索引里。

### 1.4 两个 skill 动作

除了环境命令，模型还可以执行两个 skill 动作。它们和环境命令格式相同，都占一步：

- **`load_skill S<k>`（加载）**：读取索引里第 k 条 skill 的正文，正文作为下一条观测返回。任何一轮都可以加载。
- **`write_skill`（写）**：把经验写成一条新 skill，在 `Action: write_skill` 后面接三个字段：

  ```
  Thought: <值得记下的是什么>
  Action: write_skill
  Name: <名字>
  When to use: <一句话>
  Procedure:
  1. <步骤>
  2. <步骤>
  ```

  写完后这一局继续。什么时候写、写不写、写几条，都由模型自己决定。

一条轨迹里，`write_skill` 那一轮模型生成的全部 token（`Thought` 行、`Action: write_skill` 和后面的三个字段）叫**写作 token**。§3 会给写作 token 单独的 reward。

### 1.5 训练单元

数据集的一行是一个**训练单元**：一组同题型、互不相同的题 $(A;\ B_1, \dots, B_m)$。

- **A（源题）**：A 上写出的 skill 会被拿去检验。
- **$B_1, \dots, B_m$（检验题）**：用来检验 A 上写出的 skill。

所有题的 prompt 完全相同，两个动作在每道题上都可以用，模型分不出自己在做 A 还是检验题。区别只在 harness 怎么处理：只有 A 上写出的 skill 会被检验；检验题上写出的 skill 直接丢弃，那一轮只是一个普通动作（占一步），否则检验题上写出的 skill 又需要自己的检验题。

ALFWorld 一局在任务完成时立即结束，而完成之后正是最清楚"什么管用"的时候。所以任务完成后会多给模型一轮：可以写 skill，也可以回复 `Action: done` 结束。这一轮不强制写。

A 上写出的、准备拿去检验的新 skill 叫**候选**。

### 1.6 组

一个训练单元在 rollout 时展开成若干个**组**。一个组是"同一道题 × 同一份索引"的若干条轨迹，它同时是一个 GRPO 比较组：组内轨迹的任务得分互相比较，算出 advantage。各种组的作用在 §3 和 §4 里逐一引入。

### 1.7 符号表

| 符号 | 含义 | 参考默认值 |
|---|---|---|
| m | 每个训练单元的检验题数 | 2 |
| G | A 上采样的轨迹数，即 verl 的 `rollout.n` | 8 |
| M | 每道检验题上对照组的轨迹数（§3.2） | 4 |
| K | 每个候选在所有检验题上的实验轨迹总数，每道检验题 K/m 条（§3.2） | 8 |
| N | 索引里的 skill 条数 | 4 |
| τ | 影响力阈值（§3.2） | 0.1 |
| κ | 功劳的收缩常数（§3.3） | 1 |
| β | 写作 token 上功劳项的系数（§3.4） | 1 |

---

## 2. Benchmark 与训练集

### 2.1 为什么选 ALFWorld（文本版）

| 要求 | ALFWorld 的情况 |
|---|---|
| 程序化验证 | 成功与否由模拟器的目标谓词判定，不需要 LLM judge |
| 有题型结构，同题型解法相通、答案不同 | 6 个题型（pick / look / clean / heat / cool / pick2），同题型流程相同，物体、容器、房间不同 |
| 有需要"从经验里学到"的知识 | 动作语法（本版本放置用 `move X to Y` 而不是 `put`）、哪个电器能加热、先走到再交互、关着的容器要先打开。这些题面里没有，只能试出来，试出来后对同题型通用 |
| rollout 便宜 | `pip install alfworld` 加约 1.5 GB 数据，纯 CPU，无 Java / Docker；实测单局初始化 0.4–2.5 s |
| 规模够 | 3553 个训练游戏，官方自带 seen / unseen 房间的验证集 |

考虑过但是落选的其他benchmark：ScienceWorld（要 Java、回合长、小模型冷启动几乎为 0）；AppWorld（代码形态的 skill 很合适，但需要大模型）；Game of 24 这类（一条 skill 解决所有题，没有"积累"可言）。

### 2.2 训练集：一行是一个训练单元

构造脚本：[`examples/data_preprocess/alfworld_skill_chains.py`](../../examples/data_preprocess/alfworld_skill_chains.py)。一行的格式（`source` 是 A，`probes` 是检验题）：

```jsonc
{"data_source": "alfworld/train/pick", "agent_name": "skill_agent",
 "extra_info": {
   "family": "pick", "phase": 0,
   "source": {"task_id": "...", "family": "pick",                       // A
              "payload": {"gamefile": "...", "goal": "put a candle in drawer", ...}},
   "probes": [{"task_id": "...", "family": "pick", "payload": {...}},   // B_1
              {"task_id": "...", "family": "pick", "payload": {...}}]}} // B_2
```

**检验题怎么选。** 检验题与 A 同题型，优先选物体不同且房间不同的题，并且永远不与 A 共享 `(物体, 容器, 房间)` 三元组。流程可以迁移，答案不能，所以 skill 里背下 A 的答案（"杯子在 cabinet 3"）在检验题上没有用。生成后校验过：2207 行里跨题型的检验题 0 个，重复的题 0 个。

**切分（按题型留出）：**

| 集合 | 内容 | 行数 | 用途 |
|---|---|---|---|
| `train` | pick 790 / look 308 / clean 650 / heat 459 | 2207 个训练单元 | 训练 |
| `val_seen` | 训练题型，训练见过的房间 | 91 | 分布内 |
| `val_unseen` | 训练题型，没见过的房间 | 96 | 房间泛化 |
| `val_heldout` | cool 46 / pick2 41 | 87 | 题型迁移：这两类从不参与训练，也从不产生 skill |
| `test_heldout_large` | 训练集里没用到的 cool / pick2 游戏各 100 | 200 | 同上，样本更多 |

留出 cool 和 pick2 是有意挑的：cool 的流程是 heat 换一个电器，pick2 是 pick 做两遍。

**顺序（必须 `data.shuffle=False`）。** 一个训练步按顺序读 parquet 里连续的 8 行（`data.train_batch_size = 8`，即 8 个训练单元），所以行的顺序决定了每一步做哪些题型。顺序分三段（步数是构造脚本的默认参数，可调）：

1. 前 10(参考值) 步：每步只有 pick 和 look 两个简单题型，各 4 个训练单元。行里的 `phase` 字段记为 0，下同。
2. 接下来 40(参考值) 步：每步从四个训练题型里取 2 个，各 4 个训练单元，按步轮换。
3. 剩下的行：四个题型均匀混合。

每步只放 2 个题型并轮换的原因：库在步末才提交，第 t 步写出的 skill 最早在第 t+1 步被读到。如果一个题型隔很多步才再出现，一条 skill 从写出到被用上、被重新测量的间隔就会拉长。

---

## 3. Reward 设计与归因（核心）

### 3.1 难点

写 skill 的 reward 是延迟的、间接的：写 skill 的那条轨迹本身不会因为写了它而得分更高，它的价值体现在别的题上。有三件事要同时解决：

1. **延迟**：价值在未来的题上才显现，而 on-policy 训练里轨迹用完就丢。
2. **间接**：未来那道题做对了，可能是 skill 的功劳，也可能是题简单，或者模型本来就会。
3. **没被用**：skill 出现在索引里不代表被用了。没加载，或者加载后没照着做，这时比较分数只是在比较噪声。

### 3.2 当步检验：源组、对照组、实验组，以及使用权重

对每个训练单元，先在 A 上跑一个组：

- **源组**：A 的索引是从旧库检索出的 N 条，跑 G 条轨迹。每条轨迹得到 A 的任务得分，其中一部分轨迹写出了 skill。这些 skill 去重后成为候选（每个训练单元最多检验 4 个候选，超出的不检验）。

然后对每道检验题 $B_j$ 跑：

- **对照组**：$B_j$ 的索引是从旧库检索出的 N 条，跑 M 条轨迹。M 条的平均任务得分叫 $B_j$ 的**基线分** $b_j$，同一训练单元的所有候选共用。
- **实验组**（每个候选 g 一个）：索引是旧库检索结果加上 g，跑 K/m 条轨迹。g 直接放进索引、位置随机，不经过检索，避免检索失误被算到写 skill 的头上。加载不加载 g 仍由模型自己决定。

这一步解决了"延迟"：检验和写作发生在同一个训练步，用的是同一版模型。对照组和实验组只差 g 这一条，解决了"间接"。剩下"没被用"：实验组里没加载 g 的轨迹，它们和对照组的得分差与 g 无关。为此对实验组的每条轨迹 k 算一个**使用权重** $u_k \in [0, 1]$：

- 轨迹 k 没有加载 g：$u_k = 0$。
- 加载了：计算 g 的**影响力**。保持这条轨迹的 token 序列不变，把 g 的正文换成空，重新算一遍"加载之后模型生成的每个 token"的对数概率。影响力是这些 token 在"有正文"和"正文为空"两种上下文下对数概率之差的平均值。然后 $u_k = \mathrm{clip}(\text{影响力} / \tau,\ 0,\ 1)$：影响力达到 τ 时权重为 1。

影响力的含义：轨迹是在有正文的条件下采样的，所以影响力的期望是"有正文的策略"和"无正文的策略"之间的 KL 散度，衡量的是行为被正文改变了多少。加载后没照着做，两次概率几乎一样，影响力约为 0。计算只需要对已有轨迹做前向（真实轨迹一次，每条被测 skill 的空白版一次），不需要额外采样。

### 3.3 功劳

候选 g 的**功劳**：

$$
\text{功劳}(g) = \frac{\sum_k u_k \,\bigl(r_k - b_{j(k)}\bigr)}{\sum_k u_k + \kappa}
$$

求和范围是 g 在所有检验题上的实验组轨迹。$r_k$ 是轨迹 k 的任务得分，$j(k)$ 是它做的那道检验题，$b_{j(k)}$ 是那道题的基线分。

| 部分 | 含义 |
|---|---|
| $r_k - b_{j(k)}$ | 比"不带 g 做同一道题"多得的分。每道题减自己的基线，题的难度差异被抵消 |
| $u_k$ | 这条轨迹真正用了 g 的程度 |
| $\kappa$ | 收缩常数：用了 g 的轨迹越少，功劳越往 0 收缩；一次都没用就是 0 |

各种情况下的结果：

| 情况 | 功劳 |
|---|---|
| 所有轨迹都没加载 | 0 |
| 加载了但没照着做 | 约 0 |
| 照着做了，分数比基线高 | 正 |
| 照着做了，和基线差不多 | 约 0 |
| 旧库里已有等价的 skill | 设计意图是约 0；用脚本策略（手写规则代替模型、skill 的真实价值已知，见 VALIDATION §3）检验时实测偏正，原因和数字见 REFLECTION §1 |
| 照着做了，反而更差 | 负 |
| 只在一道检验题上有用 | 被拉低 |

实现在 [`credit.py`](../../verl/experimental/skill_loop/credit.py)，`test_credit_on_cpu.py` 里有一个手算的例子（两道题、四条轨迹，功劳 0.98 / 3.1）。

### 3.4 每类 token 的 reward 和优化组

参与训练的优化组（组内做 GRPO 归一化）有四种：

| token | reward | 在哪些值之间归一化 |
|---|---|---|
| 源组轨迹的 token（写作 token 除外） | A 的任务得分 | 源组的 G 条轨迹 |
| 源组轨迹里的写作 token | A 的任务得分的 advantage + β × 功劳的 advantage | 两项各自在源组的 G 条轨迹之间归一化，再相加 |
| 对照组轨迹的 token | 自己的任务得分 | 同一道检验题的对照组 M 条 |
| 实验组轨迹的 token | 自己的任务得分 | 同一道检验题、同一个候选的实验组 K/m 条 |

对照组和实验组虽然是为了测功劳而跑的，但它们也是模型在真实题目上的正常轨迹，加载 skill、照着做的行为能从中学到，所以照常训练。§4 里还有两种组（复测组、合并组），只用来给库里的 skill 打分，轨迹记录下来但不进训练：它们的索引是为了测量人为构造的（去掉或强行塞进某条 skill），不是模型平时会看到的输入。

几点说明：

- **功劳作用在整段写作 token 上**，包括 `Action: write_skill` 本身。这样"决定写"这个动作也被训练，而不只是"写什么"。
- **没写的轨迹，功劳按 0 参与归一化。** 例如源组 4 条轨迹里两条写了，功劳是 +0.3 和 −0.1，另外两条没写：均值 0.05，两段写作 token 的功劳 advantage 是 +0.25 和 −0.15。只有一条轨迹写了时，advantage 也不是 0。模型学到的是有值得写的东西时才写。写了但因为超过 4 个候选而没有被检验的 skill，不参与归一化，功劳项为 0。
- **A 的得分和功劳分别归一化再相加**，避免 0/1 量级的任务得分压过功劳。A 的得分项在写作 token 上的系数是配置 `write_task_coef`（默认 1），β 是配置 `write_credit_coef`。
- **检验题上没有"使用奖励"。** 用 skill 有好处，得分自然体现；乱加载被带偏，得分自然惩罚。
- **按"哪道题 + 哪份索引"分组。** 不同索引的轨迹混在一起归一化，轨迹会因为"拿到了好的索引"被奖励，而这不是它自己的动作造成的。
- **写 skill 的 reward 只来自当步的检验**，不做跨步回传（那时模型已经变了）。长期价值交给库的维护（§4）。
- 一条轨迹写了多条 skill 时，每条各自是一个候选、各自测功劳。几条轨迹写出重复的 skill 时，去重成一个候选，它们共用这个候选的功劳。

### 3.5 每个设计防住了什么

| 设计 | 防住的问题 |
|---|---|
| 和旧库比，而不是和"什么都不带"比 | 重复写已有的 skill 也能拿功劳（只防住了一部分，见 REFLECTION §1） |
| 多道检验题，每道减自己的基线 | 一道题的偶然性；只贴合某一道题细节的 skill |
| κ 收缩 | 只有一两条轨迹用到时，被偶然结果误判 |
| 加载 + 影响力构成使用权重 | skill 被忽略时，估计变成噪声 |
| 检验题不给使用奖励 | 为刷分乱加载 skill |
| A 和检验题是同题型的不同题 | skill 里直接写答案 |
| 候选直接放进实验组的索引 | 检索失误被算成写 skill 的错 |
| 写作 token 单独加功劳项 | skill 的功劳被摊到整条轨迹上 |
| 索引里 `name` 和 `when_to_use` 按词数截断 | 把正文塞进永远可见的那两行 |
| 正文按词数截断 | 靠写长拿分 |
| skill 不出现在写出它的那道题的索引里 | skill 当成答案缓存 |

---

## 4. Skill 库的维护

### 4.1 什么算好 skill

"好"的定义就是功劳：在同题型的其他题上，被模型加载并照着做之后，比只带旧库多得了多少分。这个定义同时要求三件事：`when_to_use` 写得让模型在该用的时候愿意加载；正文真的改变了模型的行为；改变之后得分更高，而且旧库里还没有等价的东西。

库里每条 skill 记录：来源题型、入库时间、上次测量的步数、**效用**（最近一次测出的功劳）、被加载次数。检索出的 N 条在索引里按效用从高到低排列。

### 4.2 原则

准入用什么估计方式，维护时就用同一个方式再测一次。下面的入库、复测、合并都用 §3.3 的功劳公式，区别只在于对照组和实验组的索引怎么构造。

### 4.3 入库

功劳为正的候选在步末入库。和库里已有的 skill 很像（`when_to_use + procedure` 的向量余弦相似度 ≥ 0.9）时，只保留功劳更高的那条。

### 4.4 复测和删除

一条 skill 距上次测量超过 `retest_every` 步（默认 2）后可以被复测。复测搭在同题型的训练单元上，在每道检验题上跑两个**复测组**：

- 索引是"旧库检索结果去掉这条 skill"；
- 索引是"旧库检索结果带上这条 skill"（没被检索到就直接放进去）。

两组之一就是这个训练单元本来要跑的对照组，所以每道检验题只多跑一个组（4 条轨迹）。套同一个功劳公式，结果更新为这条 skill 的效用。优先复测很久没测的、被加载得多的、加载后结果常不好的。

复测结果 ≤ 0.02（约等于 0 或为负）记一次；连续两次就删除。约等于 0 对应的情况是内容已被模型学进参数，或者被别的 skill 覆盖。要连续两次是因为单次测量的噪声和功劳本身一样大（见 REFLECTION），单次就删会让好 skill 迟早被某次坏运气删掉。

### 4.5 合并

每隔 `merge_every` 步（默认 4），把同题型里相似（相似度 ≥ 0.8）的 skill 聚成一簇，让模型写一条合并版（这次生成的 token 不参与训练）。在检验题上跑三个**合并组**，每组 4 条轨迹：索引去掉这一簇、带着这一簇、带着合并版。以"去掉这一簇"为基线，分别算这一簇和合并版的功劳；合并版的功劳不低于这一簇的功劳就替换，否则不动，8 步内不再尝试合并这一簇。

### 4.6 容量

每个题型最多 8 条 skill，超了按效用从低到高淘汰。

实现在 [`skill_library.py`](../../verl/experimental/skill_loop/skill_library.py)。

---

## 5. 一个训练步

```
对每个训练单元 (A; B_1, ..., B_m)，各单元并发：
  1. 给 A 从旧库检索索引，跑源组 G 条。每条得到 A 的任务得分，其中一部分写出了 skill
  2. 写出的 skill 去重，得到候选（最多 4 个）
  3. 向库要本步的复测对象和合并对象（同题型）
  4. 每道检验题 B_j 并发跑：对照组、每个候选的实验组、复测组、合并组
  5. 对索引里有被测 skill 的轨迹：看是否加载，算影响力，得到使用权重
  6. 算每个候选的功劳、复测 skill 的功劳、合并版和原簇的功劳
  7. 源组、对照组、实验组的轨迹按组算任务得分的 advantage；写作 token 再加功劳的 advantage
  8. 把候选和功劳、复测结果、合并结果、加载次数暂存到库
所有单元做完：做一次 GRPO 更新，然后库提交（入库、更新效用、删除、合并、容量淘汰）
```

默认规模下（G = 8，m = 2，M = 4，K = 8），一个训练单元约 8 + 2 × (4 + 4 × 候选数) 条轨迹，候选数为 4 时是 48 条。有复测时每单元多 8 条、有合并时多 24 条，这些只用于测量。

---

## 6. Harness 改造

### 6.1 默认 rollout 的数据流，以及它为什么撑不起闭环

```
dataloader ──prompt(+uid)──▶ AgentLoopManagerTQ ──chunk──▶ AgentLoopWorkerTQ × 多个 (Ray actor，互不共享状态)
                                                              │ _run_prompt: 对每个 prompt 并发起 n 个 session
                                                              │   每个 session: hydra 新建一个 AgentLoop → run() → 算 reward
                                                              ▼
                                               TransferQueue  key = {uid}_{session}_{index}
                                                              │
trainer: ReplayBuffer.sample ─▶ old_log_prob ─▶ GRPO advantage(按 uid 分组) ─▶ update ─▶ kv_clear
```

| # | 默认实现的事实 | 对 skill 闭环意味着什么 |
|---|---|---|
| 1 | 每条轨迹新建一个 AgentLoop 实例；worker 是互相独立的 Ray actor | 没有跨轨迹、跨 worker、跨步的状态，无处存 skill 库 |
| 2 | prompt 在 dataset 里就定了，rollout 只是消费它 | 没有把之前 rollout 的产物放进之后 prompt 的通道 |
| 3 | 一个 prompt 的 n 个 session 是并发的、互不可见的独立任务 | 做不了"先做 A、再拿 A 上写出的 skill 去做检验题"这种有依赖的阶段 |
| 4 | 一个 session 一个 reward | 同一条轨迹里的写作 token 无法有不同的 reward |
| 5 | GRPO 按 `uid` 分组，一个 prompt 的所有 session 是一个比较组 | 做不同题、带不同索引的轨迹被迫共用一个 baseline |
| 6 | 生成接口只返回采样 token 的 logprob | 影响力需要"给定 token 序列的 logprob" |

一句话：verl 的 rollout 假设是"无状态 × 样本独立 × 一条轨迹一个 reward × 一个 prompt 一个比较组"，skill 闭环在四点上都相反。

### 6.2 改造后的结构

```
                           ┌────────────────────────────────────────────┐
                           │ SkillLibraryActor (named Ray actor)        │  唯一的跨轨迹 / 跨步状态
                           │  retrieve │ retest_plan │ merge_plan       │
                           │  propose │ report_loads/retest/merge       │
                           │  commit(step) ◀── trainer                  │
                           └──────▲───────────────▲─────────────────────┘
                   读已提交版本    │               │ 暂存测量结果
dataloader(训练单元) ─▶ SkillLoopManagerTQ ─▶ SkillLoopWorkerTQ._run_prompt  (训练单元级编排)
                                             ├─ 源组：SkillAgentLoop.solve ×G（可 load_skill / write_skill）
                                             ├─ 候选去重、取复测 / 合并计划
                                             ├─ 各检验题上的组并发：solve ×M、×K/m …（可 load_skill；测影响力）
                                             ├─ credit.py：使用权重 → 功劳 → 写作 token 的 advantage
                                             └─ _put_rows：每行 = 一条参与训练的轨迹，自带 reward、组名、write_mask / write_adv
                                                        ▼
                                                 TransferQueue
                                                        ▼
SkillEvolveTrainerSync: sample ─▶ old_log_prob ─▶ 按 (uid, 组) 的 advantage + 写作 token 的功劳项 ─▶ update
                        on_step_end: library.commit(step) + 落盘快照
```

### 6.3 改了哪些文件

**对 verl 已有文件的修改一共三处，都是开一个扩展点，不改默认行为**（`git diff a4e0dd2 -- verl/trainer`）。

| 文件 | 改动 | 为什么 |
|---|---|---|
| `verl/trainer/ppo/v1/agent_loop_tq.py` | 把 `@ray.remote class AgentLoopWorkerTQ` 拆成普通类 `AgentLoopWorkerTQBase` + `AgentLoopWorkerTQ = ray.remote(...)`；`AgentLoopManagerTQ` 增加类属性 `agent_loop_worker_cls` | Ray actor 类不能被继承。训练单元级编排要重写 `_run_prompt`，必须能继承 worker |
| `verl/trainer/ppo/v1/trainer_base.py` | 把 `_compute_advantage` 里写死的 advantage 调用抽成可覆写方法 `_compute_advantage_estimates(data, batch, metrics)` | 默认的"每 session 一个 reward、按 uid 比较"写死在函数体中间 |
| `verl/trainer/main_ppo.py` | 支持 `+trainer.v1.trainer_class=<FQN>` 覆盖 trainer 类 | trainer 里大量 `self.trainer_mode == "sync"` 判断，不能靠注册一个新 mode 名来换类 |

**新增文件：**

| 文件 | 行数 | 职责 |
|---|---|---|
| `verl/experimental/skill_loop/credit.py` | 174 | 影响力、使用权重、功劳、写作 token 的 advantage。纯 Python |
| `…/skill_library.py` | 472 | 库：检索、复测 / 合并计划、暂存、提交（入库 / 去重 / 删除 / 合并 / 容量）、持久化 |
| `…/embedding.py` | 105 | 检索和去重用的文本向量（哈希词袋，或 HF 句向量模型） |
| `…/skill_library_actor.py` | 91 | 把库包成 Ray named actor |
| `…/skill_agent_loop.py` | 390 | `SkillAgentLoop(AgentLoopBase)`：`solve()` 一局多轮交互，处理 `load_skill` / `write_skill`，记录写作 token 的位置，对被测 skill 算影响力；token-in-token-out，复用 verl 的 Continuous Token 构造器 |
| `…/skill_loop_tq.py` | 732 | `SkillLoopWorkerTQ`：重写 `_run_prompt`，把一个训练单元展开成各个组、算功劳、分配 reward；`_put_rows` 按轨迹写 TransferQueue；`SkillLoopManagerTQ` 创建库 actor |
| `…/skill_advantage.py` | 98 | 按 `(uid, 组)` 的 GRPO advantage，加上写作 token 的功劳项 |
| `…/skill_trainer.py` | 274 | `SkillEvolveTrainerSync(PPOTrainerSync)`：覆写 advantage 钩子；`on_step_end` 里提交库、记指标、落快照 |
| `…/rollout_only.py` | 85 | 不带模型训练引擎的 rollout 驱动：走同一套 trainer 数据通路，只是不做梯度更新 |
| `…/prompts.py` | 207 | prompt、索引渲染、动作 / 加载 / 写作的解析 |
| `…/envs/base.py`, `envs/alfworld_env.py` | 196 | 环境接口 + ALFWorld 适配。换 benchmark 只需要再写一个 `make_env` |
| `verl/trainer/config/skill_evolve/default.yaml` | — | Hydra 配置组，`+skill_evolve=default` 加载，每个参数有注释 |
| `examples/data_preprocess/alfworld_skill_chains.py` | 300 | 训练集构造 |
| `examples/skill_evolve/*` | — | 启动脚本、轨迹报告渲染（`render_trajectories.py`）、功劳估计的检验（`analyze_credit.py`）、两种可替换的推理后端（脚本策略、外部 `vllm serve` 的 HTTP 客户端） |
| `tests/experimental/skill_loop/*` | 1209 | 52 个 CPU 测试 |

### 6.4 接口怎么对接

**启用（三个开关，其余都是普通 verl 配置）：**

```bash
+skill_evolve=default
+trainer.v1.trainer_class=verl.experimental.skill_loop.skill_trainer.SkillEvolveTrainerSync
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.skill_loop.skill_loop_tq.SkillLoopManagerTQ
```

`skill_evolve.enable=false` 或样本的 `agent_name` 不是 `skill_agent` 时，worker 原样回退到上游的 `_run_prompt`。

**与 TransferQueue 的接口**（沿用上游，只加不改）：

- key 仍是 `{uid}_{session_id}_{index}`，一条参与训练的轨迹一行；prompt 的 `pending → running → finished / failure` 状态机不变，所以 ReplayBuffer、失败处理、padding 都不用动。一个训练单元的所有行用一次 batched put 写入，再标 `finished`：一个训练单元要么完整出现要么不出现。
- 每行的 `rm_scores` 是它自己的任务得分。
- 每行多四列：`adv_group`（组名，如 `source`、`check0/control`、`check1/new2`）、`write_mask` 和 `write_adv`（token 级，和 `response_mask` 对齐：写作 token 上分别为 1 和 β × 功劳的 advantage）、`skill_meta`（json，只用于指标）。
- trainer 的 advantage 钩子：先把任务得分在 `(uid, adv_group)` 内归一化得到 advantage A，在 `write_mask` 为 1 的 token 上换成 `write_task_coef × A + write_adv`。
- `rollout.n` 是源组的轨迹数 G。一个 prompt 产出的行数不固定，行数不整除 mini-batch 时由上游已有的 padding 逻辑补齐。

**影响力需要的打分接口。** verl 的 vLLM server 已经支持在采样参数里传 `prompt_logprobs`，并把结果放在 `extra_fields["prompt_logprobs"]` 里。agent loop 用它对一段给定的 token 序列做一次前向（`max_tokens=1`），不需要改 server。rollout-only 驱动用的两个推理后端实现了同一个约定。

**与库的接口：**

| 调用方 | 调用 | 时机 |
|---|---|---|
| worker | `retrieve(query, task_id)` | A 和每道检验题各一次；只读已提交版本 |
| worker | `cluster_texts(texts)` | 源组做完后，给写出的 skill 去重 |
| worker | `retest_plan(family, step)`、`merge_plan(family, step)` | 源组做完后；同一步内一条 skill 只会被领走一次 |
| worker | `propose` / `report_retest` / `report_merge` / `report_loads` | 训练单元做完 |
| trainer | `commit(step)` → `save(path)` | `on_step_end` |

**为什么提交放在 `on_step_end`：** 这让库和权重有同样的版本语义，第 t 步的所有 rollout 读的是权重 $\theta_{t-1}$ 和库 $L_{t-1}$。同一个组的轨迹看到同一份索引是组内比较成立的前提；库在一步之内不变，也让一步之内的数据保持 on-policy（库是策略上下文的一部分）。

**验证路径：** `validate=True` 时每条轨迹只做题：从冻结的库检索索引，prompt 和训练时相同，写出的 skill 丢弃，不暂存任何东西。`skill_evolve.val_use_skills=false` 可以关掉库，用来做同一 checkpoint 带库和不带库的对比。

