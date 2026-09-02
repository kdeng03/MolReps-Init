# Molecule Representation Learning Project Plan

## 1. 项目目标

这项工作的核心目标，是验证大模型是否真的能够“理解分子表示”，并且比较不同分子表示形式在训练和推理中的学习难度、泛化能力和可迁移性。

当前项目的主线是 **文本侧 SFT / LoRA SFT**，OCR 只作为 **VLM 的 cold-start ablation** 来做对比，不作为主任务混进统一训练集里。这样做的目的，是把“LLM 对文本表示的学习能力”和“VLM 加上 OCR 之后是否带来迁移收益”分开分析，避免实验结论混淆。

我准备围绕以下问题展开实验：

- 机器是否能把不同分子表示（SMILES / SELFIES / DeepSMILES / InChI / IUPAC / CML）相互映射？
- 哪种表示最容易被模型学会？
- 训练时是否需要同时混合多种任务，才能获得更强泛化？
- 对“随机化版本 → canonical 版本”的归一化任务，模型是否真的能学到结构级理解，而不是只记忆字符串模式？
- 如果把 OCR 作为 VLM 的 cold start，是否能对后续文本任务带来正迁移？

---

## 2. 研究假设

我先把这个项目定义为一个“representation understanding + representation conversion”任务，而不是单纯做一类 QA。

主要假设如下：

1. 语言模型可以通过监督微调学习分子表示的结构对齐关系；
2. 训练数据中同时包含“翻译”“理解”“归一化”三类任务，能比单一任务得到更强的表示理解能力；
3. 不同表示形式在可学习性上并不相同：
   - SMILES 可能更容易被直接学习；
   - SELFIES / DeepSMILES 可能更适合做规范化任务；
   - InChI / IUPAC / CML 更偏向“结构语义理解”。

---

## 3. 实验路线

### 3.1 数据构建

第一阶段先把原始分子表示数据生成出来，作为后续所有实验的基础数据源。

我准备生成两层数据：

- Stage A: raw representation data
  - 记录每个分子的基础事实：canonical SMILES、canonical SELFIES、canonical DeepSMILES、InChI、IUPAC、CML 等；
  - 同时生成随机变体（random SMILES / random SELFIES / random DeepSMILES）。

- Stage B: SFT / QA-style task data
  - 生成三类任务：
    - translation：canonical representation 间的互相转换；
    - understanding：从 IUPAC / CML 反推 canonical 表示；
    - normalization：把随机变体规范化为 canonical 形式。

这一步的产物会是：

- 原始表示数据集：用于后续复用和扩展；
- 可直接训练的任务数据集：用于做监督微调。

---

### 3.2 训练实验

我准备做一组比较清晰的训练实验，重点观察“任务类型”和“训练混合方式”对模型效果的影响。

实验设置会尽量保持简洁：

- 以 molecule 为单位做 split，而不是按样本做 split，避免同一分子的不同表示同时出现在 train/test 中造成泄漏；
- 先用 100 个 molecule 做小规模验证，其中 80 个 molecule 作为 train，20 个 molecule 作为 test；
- 文本任务采用 N = 2 的随机变体配置，优先看 base model 和 LoRA SFT 的趋势；
- OCR 单独走 VLM 分支，只作为对比实验，不与文本样本硬混在一个训练集里。

#### A. Baseline

- 直接使用预训练基础模型，不做分子表示专门训练；
- 作为参照系，观察模型原生能力有多强。

这里会保留两个基线：

- Base LLM zero-shot：只看文本任务；
- Base VLM zero-shot：看图像 OCR 任务，以及文本任务是否会被视觉结构影响。

#### B. 单任务训练

分别训练以下几个版本：

- translation-only：只学 representation translation；
- understanding-only：只学 IUPAC / CML 到 canonical 的映射；
- normalization-only：只学随机变体归一化。

这一步主要回答：

- 哪种任务最容易让模型学会表示转换？
- 哪种任务最能提升结构理解能力？

#### C. 混合任务训练

再训练几个混合版本：

- translation + normalization；
- understanding + normalization；
- all tasks（translation + understanding + normalization）。

这一步主要回答：

- 多任务联合训练是否比单任务更有效？
- 是否存在“任务互补”效应？

#### D. Ablation 训练

为了验证某些能力是不是“被额外任务带出来的”，我会做消融：

- 去掉 IUPAC 理解；
- 去掉 CML 理解；
- 只保留 intra normalization；
- 只保留 cross normalization；
- 只保留一部分 representation 类型。

这一步的核心目的是看：

- 结构理解能力是否依赖更多语义任务；
- 归一化任务是否能提升更广义的 representation 对齐能力。

#### E. OCR cold-start ablation

OCR 这条线单独作为 VLM 的对照实验，不和文本任务混成一个统一训练集，而是走两阶段：

- Phase 1：只做 OCR SFT，让 VLM 学会从 2D 分子图输出 canonical 表示；
- Phase 2：接文本 SFT，只看 OCR 是否对文本任务有迁移，或者是否会发生遗忘。

这一组实验主要回答两个问题：

- OCR 作为 cold start 是否能帮助后续文本 SFT；
- VLM 做完文本 SFT 后，OCR 能力是否会下降。

---

## 4. 产出物

### 4.1 数据产物

- 生成的 raw molecule representation 数据集；
- 生成的 QA / SFT 数据集；
- 不同任务子集的数据（translation / understanding / normalization / mixed）。

### 4.2 模型产物

我预期会产出以下模型或适配器：

- 基础模型的零-shot 结果基线；
- 单任务微调模型；
- 多任务混合微调模型；
- 可能的 LoRA / adapter 版本；
- 最终选出来的一个“分子表示转换专家模型”。

### 4.3 结果产物

- 每个实验的结果表格；
- 不同任务和不同表示上的准确率 / 规范化成功率 / 变体鲁棒性报告；
- 最终结论摘要和实验分析文档。

---

## 5. 评测方式

我准备从“字符串级”和“结构级”两个角度评估模型。

### 5.1 字符串级评估

- exact match：输出是否与目标 canonical 表示完全一致；
- normalized string accuracy：对等价字符串做规范化后再比较；
- 任务级准确率：按 translation / understanding / normalization 分别统计。

对于 OCR 任务，会额外单独报告 image-to-text 的 exact match 和 RDKit match；对于文本任务，会单独按 LLM 和 VLM 两条线分别汇报，避免把纯文本能力和视觉分支能力混在一起看。

### 5.2 结构级评估

- 对生成结果做 RDKit 解析，检查是否能正确生成可解析的分子对象；
- 对不同随机变体输入，观察模型是否能稳定恢复到同一个 canonical 目标；
- 比较模型在“同分子不同表示输入”下的稳定性。

---

## 6. 主要对比内容

我计划重点对比以下几个维度：

1. 表示形式对比
   - SMILES vs SELFIES vs DeepSMILES vs InChI；
   - 看哪一种更容易被学习和泛化。

2. 任务类型对比
   - translation vs understanding vs normalization；
   - 看哪类任务更能提升模型的结构理解。

3. 训练混合方式对比
   - single-task vs multi-task；
   - 看联合训练是否比单任务更好。

4. 基线对比
   - base model vs SFT model；
   - 看微调是否真的带来了能力提升。

5. 模态对比
   - LLM pure text SFT vs VLM pure text SFT；
   - 看视觉分支是否会影响纯文本任务，或者只是在 OCR 上提供额外收益。

6. 代表性样本对比
   - 对随机变体输入，观察模型能否稳定输出同一 canonical 结果；
   - 对复杂分子和简单分子分别统计性能。

---

## 7. 我预期得到的结论

如果实验顺利，我希望能得到以下几类结论：

- 哪种分子表示最适合做大模型学习的目标表示；
- 哪类任务最有利于提升模型的表示理解能力；
- 训练数据中混入归一化任务是否能显著提升模型对“等价结构”的识别能力；
- 模型是否真的学到了“结构级理解”，而不是只学会了某些字符串模式；
- 哪些任务组合最值得做成最终训练集。

最终我会把结论整理成一份简短结论报告，包括：

- 最强模型是什么；
- 最有价值的数据混合是什么；
- 最值得继续扩展的实验方向是什么。

在这个项目里，结果呈现会优先围绕下面几张表来组织：

- Main Results：LLM zero-shot、LLM text SFT、VLM zero-shot、VLM text SFT、VLM OCR SFT、VLM OCR → text SFT 的主对比；
- Per-task Breakdown：按 translation / understanding / normalization 分任务报告；
- OCR Results：单独看 VLM 的 OCR 能力和遗忘情况；
- Representation Heatmap：不同 representation pair 的难度差异。

---

## 8. 已确认的实验设定

当前这版实验设定已经基本明确，后续实现时可以直接按下面的配置推进：

1. 主任务是文本 SFT / LoRA SFT，作为对 LLM 能力的主评估。

2. OCR 只作为 VLM cold-start ablation，单独训练和单独评估，不和文本样本混在统一训练集里。

3. 数据 split 按 molecule-level 进行，默认 80/20；同一 molecule 的不同表示不会跨 train/test。

4. 文本任务优先用 N = 2 的随机变体配置，先看 base vs LoRA SFT 的趋势，再决定是否扩展 N。

5. 指标会同时看 exact match 和 RDKit / structure-level match，并且按任务拆开报告。

---

## 9. 当前执行顺序建议

我建议按下面顺序推进：

1. 先把 raw representation 数据生成完；
2. 再生成任务型 SFT 数据，并固定 molecule-level split；
3. 先跑文本侧 baseline：Base LLM、LLM text SFT、VLM text SFT；
4. 再跑 OCR 分支：Base VLM、VLM OCR SFT、VLM OCR → text SFT；
5. 最后整理 main results、per-task breakdown、OCR ablation 和 heatmap。

这样做的好处是：

- 先把数据链路跑通；
- 再验证模型是否真的能学到这些能力；
- 避免一开始就把实验做得太大、太散。
