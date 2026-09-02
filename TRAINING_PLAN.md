# Molecule Representation Learning Training Plan

## 1. 项目目标

验证大模型是否真的能够"理解分子表示"，并比较不同分子表示形式在训练和推理中的学习难度、泛化能力和可迁移性。

**主线**: 文本侧 SFT / LoRA SFT  
**OCR**: 仅作为 VLM 的 cold-start ablation，**不与文本任务混训**（HF datasets 不支持 image + text 混合训练）

---

## 2. 任务定义

### 2.1 文本任务（Text → Text）

| 任务类别 | Input | Output | 数量/molecule (N=2) |
|---------|-------|--------|---------------------|
| **translation** | can_smiles/selfies/deepsmiles/inchi | can_smiles/selfies/deepsmiles/inchi | 12 |
| **understanding_iupac** | iupac | can_smiles/selfies/deepsmiles/inchi | 4 |
| **understanding_cml** | cml | can_smiles/selfies/deepsmiles/inchi | 4 |
| **intra_normalization** | random_smiles/selfies/deepsmiles | can_smiles/selfies/deepsmiles | 3N = 6 |
| **cross_normalization** | random_smiles/selfies/deepsmiles | can_smiles/selfies/deepsmiles/inchi (cross-family) | 9N = 18 |
| **总计** | | | **20 + 12N = 44** |

### 2.2 OCR 任务（Image → Text）

| 子任务 | Input | Output | 数量/molecule |
|--------|-------|--------|---------------|
| ocr_smiles | mol_image | can_smiles | 1 |
| ocr_selfies | mol_image | can_selfies | 1 |
| ocr_deepsmiles | mol_image | can_deepsmiles | 1 |
| ocr_inchi | mol_image | inchi | 1 |
| **总计** | | | **4** |

---

## 3. 实验设计

### 3.1 实验矩阵（对齐 PROJECT_PLAN.md）

| 实验组 | 编号 | 实验名称 | 训练数据 | 基座模型 | 目标 |
|--------|------|---------|---------|---------|------|
| **A. Baseline** | E0-A | Base-LLM-ZeroShot | 无训练 | Qwen3-4B (LLM) | 纯文本 LLM 零-shot conversion 基线 |
| | E0-B | Base-VLM-ZeroShot | 无训练 | Qwen3-4B-I (VLM) | VLM 零-shot OCR + conversion 基线 |
| **B. 单任务** | E1 | LLM-Translation | translation only | Qwen3-4B (LLM) | 纯 translation 能力 |
| | E2 | LLM-Understanding | understanding only | Qwen3-4B (LLM) | 纯 understanding 能力 |
| | E3 | LLM-Normalization | normalization only | Qwen3-4B (LLM) | 纯 normalization 能力 |
| **C. 混合任务** | E4 | LLM-Trans+Norm | translation + normalization | Qwen3-4B (LLM) | 混合训练效果 |
| | E5 | LLM-Understand+Norm | understanding + normalization | Qwen3-4B (LLM) | 混合训练效果 |
| | E6 | LLM-All-Tasks | all tasks | Qwen3-4B (LLM) | 全任务混合效果 |
| **D. Ablation** | E7 | LLM-No-IUPAC | all - iupac_understanding | Qwen3-4B (LLM) | IUPAC 任务的作用 |
| | E8 | LLM-No-CML | all - cml_understanding | Qwen3-4B (LLM) | CML 任务的作用 |
| | E9 | LLM-Intra-Only | intra_normalization only | Qwen3-4B (LLM) | intra vs cross normalization |
| | E10 | LLM-Cross-Only | cross_normalization only | Qwen3-4B (LLM) | cross normalization 作用 |
| **E. OCR Cold-Start** | E11 | VLM-OCR-Only | OCR only | Qwen3-4B-I (VLM) | 单 OCR SFT 效果 |
| | E12 | VLM-Conversion-Only | Conversion only | Qwen3-4B-I (VLM) | 单 Conversion SFT (VLM) |
| | E13 | VLM-OCR→Conversion | OCR → Conversion (两阶段) | Qwen3-4B-I (VLM) | OCR cold start 迁移效果 |

> **注意**: OCR + Conversion 混合训练不可行，因为 HF datasets 不支持图片和文本混合在一个训练集中。OCR 仅作为两阶段 cold-start 实验。

---

### 3.2 详细实验设计

#### A. Baseline（零-shot）
- **E0-A**: Qwen3-4B 零-shot 文本 conversion
- **E0-B**: Qwen3-4B-I 零-shot OCR + 文本 conversion

#### B. 单任务训练
- **E1**: translation-only (12 条/分子) → 哪种任务最容易让模型学会表示转换？
- **E2**: understanding-only (8 条/分子) → 哪种任务最能提升结构理解能力？
- **E3**: normalization-only (24 条/分子) → 归一化任务是否能提升广义 representation 对齐能力？

#### C. 混合任务训练
- **E4**: translation + normalization (36 条/分子)
- **E5**: understanding + normalization (32 条/分子)
- **E6**: all tasks (44 条/分子) → 多任务联合是否比单任务更有效？是否存在"任务互补"效应？

#### D. Ablation 训练
- **E7**: all - iupac_understanding → 结构理解能力是否依赖 IUPAC 任务？
- **E8**: all - cml_understanding → 结构理解能力是否依赖 CML 任务？
- **E9**: intra_normalization only → intra normalization 是否足够？
- **E10**: cross_normalization only → cross normalization 是否带来额外收益？

#### E. OCR Cold-Start Ablation
- **E11**: VLM-OCR-Only → VLM 能从分子图像中读取多少信息？
- **E12**: VLM-Conversion-Only → VLM 在没有图像训练的情况下，conversion 能力如何？
- **E13**: VLM-OCR→Conversion (两阶段) → OCR 作为 cold start 是否能帮助后续文本 SFT？VLM 做完文本 SFT 后，OCR 能力是否会下降？

---

## 4. 数据准备

### 4.1 数据来源
| 数据 | 来源 | 格式 | HuggingFace Dataset |
|------|------|------|---------------------|
| mol-reps-v0 | Stage 1: 原始表示 | Parquet (mol_id + reps + mol_image) | kdeng03/mol-reps-v0 |
| mol-rep-ocr-v0 | Stage 2: OCR 任务 | JSONL (prompt + completion + image) | kdeng03/mol-rep-ocr-v0 |
| mol-rep-conversion-v0 | Stage 2: Conversion 任务 | JSONL (prompt + completion) | kdeng03/mol-rep-conversion-v0 |

### 4.2 数据规模
- **分子数量**: 100 个分子（80 train / 20 test，**按分子 split**，避免同一分子的不同表示同时出现在 train/test 中造成泄漏）
- **随机变体**: N=2
- **每个分子的任务数**: 44 条（文本）+ 4 条（OCR）
- **总训练样本**: ~3520 (80 × 44)
- **总测试样本**: ~880 (20 × 44)

### 4.3 数据组织
```
data/
├── mol-reps-raw/
│   └── train.parquet          # Stage 1: 原始表示
└── mol-conversion-qa/
    ├── ocr.parquet            # OCR 任务
    ├── translation.parquet    # Translation 任务
    ├── understanding.parquet  # Understanding 任务
    ├── normalization.parquet  # Normalization 任务
    └── all.parquet            # 所有文本任务
```

---

## 5. 训练配置

### 5.1 通用配置
| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen3-4B / Qwen3-4B-I | LLM vs VLM |
| 数据 split | 按分子 split (80 train / 20 test) | 避免泄漏 |
| 随机变体 | N=2 | 每个分子 2 个 random variants |
| 样本数/分子 | 44 (20 + 12N) | translation:12, understanding:8, intra_norm:6, cross_norm:18 |
| 总训练样本 | ~3520 (80 × 44) | 初始小规模实验 |
| 总测试样本 | ~880 (20 × 44) | 初始小规模实验 |

### 5.2 LoRA 配置（待讨论 Baseline）
当前 `grpo.sh` 中的配置：
```yaml
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj]
```

**Baseline 设置建议**:
1. **LoRA rank (`r`)**: `16` 是 4B 模型的常用起点。对于化学语法这种强规则任务，`r=8` 可能容量不足，`r=32` 在 3.5k 样本上容易过拟合。建议 baseline 用 `16`，后续可 ablate `8` vs `16`。
2. **Alpha (`α`)**: `32` (2x rank) 是标准做法，scaling factor `α/r = 2` 比较稳定。
3. **Dropout**: `0.05` 偏小。对于 3.5k 的小数据集，建议提高到 `0.1` 防止过拟合，或者保持 `0.05` 但配合 early stopping。
4. **Target Modules**: 当前配置覆盖了所有 attention 和 MLP 层。对于 baseline，建议先用 `all-linear`（或保持当前列表），确保模型有足够的参数容量学习复杂的表示映射。如果显存紧张，可先只训 `q_proj, v_proj` 做快速验证。

### 5.3 训练超参数（待讨论）
| 参数 | 建议值 | 说明 |
|------|--------|------|
| 学习率 | `1e-4` ~ `5e-5` | LoRA SFT 常用范围，小数据建议偏大一点 `1e-4` |
| Batch Size | `4` | per_device |
| Gradient Accumulation | `4` | effective batch = 16 |
| Epochs | `2` ~ `3` | 3.5k 样本 × 2 epochs ≈ 400 steps，足够收敛 |
| Max Length | `4096` | VLM 训练需要，文本任务通常用不到这么长 |
| Optimizer | `AdamW` | 默认 |
| LR Scheduler | `cosine` | 默认 |

---

## 6. 评估方案

### 6.1 评估指标
| 指标 | 说明 | 适用任务 |
|------|------|---------|
| Exact Match (EM) | 输出与目标完全匹配 | 所有任务 |
| Chemical Validity | 输出是否为有效的化学表示 (RDKit parse) | OCR, Conversion |
| Chemical Equivalence | 输出与目标是否化学等价（而非字符串匹配） | Conversion |
| Token-level F1 | 输出与目标的 token 级别 F1 | 所有任务 |

### 6.2 评估矩阵
每个实验需要评估：
1. **训练任务**: 看模型是否学会了训练的任务
2. **未训练任务 (zero-shot)**: 看能力是否泛化
3. **跨任务迁移**: 看 OCR 训练是否帮助 conversion，反之亦然

---

## 7. 实施优先级

### Phase 1: 基线实验（优先）
- [ ] E0-A: Base-LLM-ZeroShot
- [ ] E0-B: Base-VLM-ZeroShot
- [ ] E1: LLM-Translation
- [ ] E2: LLM-Understanding
- [ ] E3: LLM-Normalization

### Phase 2: 混合任务
- [ ] E4: LLM-Trans+Norm
- [ ] E5: LLM-Understand+Norm
- [ ] E6: LLM-All-Tasks

### Phase 3: Ablation + OCR
- [ ] E7-E10: Ablation 实验
- [ ] E11: VLM-OCR-Only
- [ ] E12: VLM-Conversion-Only
- [ ] E13: VLM-OCR→Conversion

---

## 8. 预期结果与假设

| 假设 | 验证方式 |
|------|---------|
| H1: 语言模型可以通过 SFT 学习分子表示的结构对齐关系 | E1/E2/E3 vs E0-A |
| H2: 多任务联合训练比单任务更有效 | E6 vs E1/E2/E3 |
| H3: SMILES 更容易被学习 | 比较各表示的 EM |
| H4: SELFIES/DeepSMILES 更适合规范化任务 | E3 vs E1 |
| H5: InChI/IUPAC/CML 更偏向结构语义理解 | E2 vs E1 |
| H6: OCR cold start 有助于后续文本 SFT | E13 vs E12 |
| H7: VLM 做完文本 SFT 后，OCR 能力会下降 | E12 的 OCR zero-shot vs E11 |

---

## 9. 后续扩展

### 9.1 RL/GRPO 阶段
在 SFT 完成后，可以考虑引入 RL 训练：
- **Reward 设计**: Exact match + Chemical validity (RDKit) + Chemical equivalence (分子指纹相似度)
- **目标**: 进一步提升 conversion 准确率和鲁棒性

### 9.2 更多任务
- 3D 结构理解
- 性质预测
- 反应预测

核心假设：
- 分子图像（2D structure image）包含丰富的结构信息，可以作为 cold start 帮助模型建立"分子视觉理解"能力
- 纯文本的 conversion 任务可以进一步提升模型对表示语义的理解
- OCR → Conversion 的两阶段训练可能比单一任务训练效果更好

---

## 2. 任务定义

### 2.1 OCR 任务（Image → Text）

**输入**: 分子 2D 结构图像  
**输出**: 多种 canonical 表示（can_smiles, can_selfies, can_deepsmiles, inchi）

| 子任务 | Input | Output | 数量/molecule |
|--------|-------|--------|---------------|
| ocr_smiles | mol_image | can_smiles | 1 |
| ocr_selfies | mol_image | can_selfies | 1 |
| ocr_deepsmiles | mol_image | can_deepsmiles | 1 |
| ocr_inchi | mol_image | inchi | 1 |

**能力目标**: 模型能从分子图像中"读取"结构信息并输出标准表示。

---

### 2.2 Conversion 任务（Text → Text）

**输入**: 一种文本表示  
**输出**: 另一种文本表示

| 子任务类别 | Input | Output | 数量/molecule |
|-----------|-------|--------|---------------|
| can_reps_translation | can_smiles/selfies/deepsmiles/inchi | can_smiles/selfies/deepsmiles/inchi | 12 |
| iupac_understanding | iupac | can_smiles/selfies/deepsmiles/inchi | 4 |
| cml_understanding | cml | can_smiles/selfies/deepsmiles/inchi | 4 |
| intra_rep_normalization | random_smiles/selfies/deepsmiles | can_smiles/selfies/deepsmiles | 6 |
| cross_rep_normalization | random_smiles/selfies/deepsmiles | can_smiles/selfies/deepsmiles/inchi (cross-family) | TBD |

**能力目标**: 模型能理解不同表示的语义，并在它们之间准确转换。

---

### 2.3 OCR + Conversion 联合任务

**输入**: 分子 2D 结构图像  
**输出**: 多种 canonical 表示（与 OCR 相同，但训练策略不同）

这个任务在数据格式上与 OCR 相同，但训练目标不同：
- OCR: 训练模型"读取"图像
- OCR+Conversion: 训练模型在"读取"图像后，进一步理解表示之间的关系

---

## 3. 实验设计

### 3.1 核心原则

根据 PROJECT_PLAN.md 的设计：
- **主线**: 文本侧 SFT / LoRA SFT
- **OCR**: 仅作为 VLM 的 cold-start ablation，**不与文本任务混训**
- **目的**: 把"LLM 对文本表示的学习能力"和"VLM + OCR 是否带来迁移收益"分开分析

> **注意**: E5/E6（OCR+Conversion 混合训练）不可行，因为 HF datasets 不支持图片和文本混合在一个训练集中。

### 3.2 实验矩阵

| 实验组 | 实验编号 | 实验名称 | 训练数据 | 基座模型 | 目标 |
|--------|---------|---------|---------|---------|------|
| **A. Baseline** | E0-A | Base-LLM-ZeroShot | 无训练 | Qwen3-4B (LLM) | 纯文本 LLM 零-shot conversion 基线 |
| | E0-B | Base-VLM-ZeroShot | 无训练 | Qwen3-4B-I (VLM) | VLM 零-shot OCR + conversion 基线 |
| **B. 单任务** | E1 | LLM-Translation | translation only | Qwen3-4B (LLM) | 纯 translation 能力 |
| | E2 | LLM-Understanding | understanding only | Qwen3-4B (LLM) | 纯 understanding 能力 |
| | E3 | LLM-Normalization | normalization only | Qwen3-4B (LLM) | 纯 normalization 能力 |
| **C. 混合任务** | E4 | LLM-Trans+Norm | translation + normalization | Qwen3-4B (LLM) | 混合训练效果 |
| | E5 | LLM-Understand+Norm | understanding + normalization | Qwen3-4B (LLM) | 混合训练效果 |
| | E6 | LLM-All-Tasks | all tasks | Qwen3-4B (LLM) | 全任务混合效果 |
| **D. Ablation** | E7 | LLM-No-IUPAC | all - iupac_understanding | Qwen3-4B (LLM) | IUPAC 任务的作用 |
| | E8 | LLM-No-CML | all - cml_understanding | Qwen3-4B (LLM) | CML 任务的作用 |
| | E9 | LLM-Intra-Only | intra_normalization only | Qwen3-4B (LLM) | intra vs cross normalization |
| | E10 | LLM-Cross-Only | cross_normalization only | Qwen3-4B (LLM) | cross normalization 作用 |
| **E. OCR Cold-Start** | E11 | VLM-OCR-Only | OCR only | Qwen3-4B-I (VLM) | 单 OCR SFT 效果 |
| | E12 | VLM-Conversion-Only | Conversion only | Qwen3-4B-I (VLM) | 单 Conversion SFT (VLM) |
| | E13 | VLM-OCR→Conversion | OCR → Conversion (两阶段) | Qwen3-4B-I (VLM) | OCR cold start 迁移效果 |

---

### 3.2 详细实验设计

#### E0-A: Base-LLM-ZeroShot

**目的**: 建立纯文本 LLM 的零-shot conversion 基线。

- **模型**: Qwen3-4B (纯文本，无 vision encoder)
- **数据**: 无训练，直接推理
- **评估**: 所有 conversion 任务的准确率

---

#### E0-B: Base-VLM-ZeroShot

**目的**: 建立 VLM 的零-shot 基线，观察视觉结构是否影响文本任务。

- **模型**: Qwen3-4B-I (VLM，带 vision encoder)
- **数据**: 无训练，直接推理
- **评估**: 
  - OCR 任务准确率（zero-shot）
  - Conversion 任务准确率（zero-shot）

---

#### E1: LLM-Conversion

**目的**: 纯文本 LLM 经过 conversion SFT 后的能力提升。

- **模型**: Qwen3-4B (纯文本)
- **数据**: 所有 conversion 任务（can_reps_translation, iupac_understanding, cml_understanding, intra/cross_normalization）
- **训练方式**: 单阶段 SFT
- **评估**: 所有 conversion 任务的准确率

---

#### E2: VLM-OCR-Only

**目的**: 评估单 OCR SFT 的效果，回答"VLM 能从分子图像中读取多少信息？"

- **模型**: Qwen3-4B-I (VLM，带 vision encoder)
- **数据**: OCR 任务（mol_image → can_smiles/selfies/deepsmiles/inchi）
- **训练方式**: 单阶段 SFT
- **评估**: 
  - OCR 任务准确率
  - Conversion 任务准确率（zero-shot，看 OCR 训练是否带来 conversion 能力提升）

---

#### E3: VLM-Conversion

**目的**: 评估单 Conversion SFT 的效果，回答"VLM 在没有图像训练的情况下，conversion 能力如何？"

- **模型**: Qwen3-4B-I (VLM，带 vision encoder)
- **数据**: 所有 conversion 任务（纯文本）
- **训练方式**: 单阶段 SFT
- **评估**: 
  - Conversion 任务准确率
  - OCR 任务准确率（zero-shot，看 conversion 训练是否带来 OCR 能力提升）

---

#### E4: VLM-OCR→Conversion (两阶段)

**目的**: 验证 OCR 作为 cold start 的效果，回答"先训练 OCR 再训练 conversion，是否比直接训练 conversion 更好？"

- **模型**: Qwen3-4B-I (VLM)
- **阶段 1**: OCR SFT（mol_image → canonical reps）
- **阶段 2**: Conversion SFT（text → text），从阶段 1 的 checkpoint 继续训练
- **评估**: 
  - OCR 任务准确率（看是否遗忘）
  - Conversion 任务准确率
  - 与 E2, E3 对比

---

## 4. 数据准备

### 4.1 数据来源

| 数据 | 来源 | 格式 | HuggingFace Dataset |
|------|------|------|---------------------|
| mol-reps-v0 | Stage 1: 原始表示 | Parquet (mol_id + reps + mol_image) | kdeng03/mol-reps-v0 |
| mol-rep-ocr-v0 | Stage 2: OCR 任务 | JSONL (prompt + completion + image) | kdeng03/mol-rep-ocr-v0 |
| mol-rep-conversion-v0 | Stage 2: Conversion 任务 | JSONL (prompt + completion) | kdeng03/mol-rep-conversion-v0 |

### 4.2 数据规模

- **分子数量**: 初始实验使用 100 个分子（80 train / 20 test，按分子 split，避免泄漏）
- **随机变体**: N=2（每个分子生成 2 个 random variants）
- **每个分子的任务数**: 20 + 12N = 20 + 24 = 44 条样本
  - translation: 12
  - understanding_iupac: 4
  - understanding_cml: 4
  - normalization_intra: 3N = 6
  - normalization_cross: 9N = 18
- **OCR 任务**: 4 条/分子（mol_image → can_smiles, can_selfies, can_deepsmiles, inchi）

### 4.3 数据组织

```
data/
├── mol-reps-raw/
│   └── train.parquet          # Stage 1: 原始表示
└── mol-conversion-qa/
    ├── ocr.parquet            # OCR 任务
    ├── translation.parquet    # Translation 任务
    ├── understanding.parquet  # Understanding 任务
    ├── normalization.parquet  # Normalization 任务
    └── all.parquet            # 所有文本任务
```

### 4.4 实验数据使用

| 实验 | 训练数据 | 备注 |
|------|---------|------|
| E0-A | 无 | Zero-shot |
| E0-B | 无 | Zero-shot |
| E1 | all.parquet | 纯文本 conversion |
| E2 | ocr.parquet | 纯 OCR |
| E3 | all.parquet | 纯文本 conversion (VLM) |
| E4 | ocr.parquet → all.parquet | 两阶段 |

---

## 5. 训练配置

### 5.1 通用配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen3-4B / Qwen3-4B-I | LLM vs VLM |
| 数据 split | 按分子 split (80 train / 20 test) | 避免同一分子的不同表示同时出现在 train/test |
| 随机变体 | N=2 | 每个分子 2 个 random variants |
| 样本数/分子 | 44 (20 + 12N) | translation:12, understanding:8, intra_norm:6, cross_norm:18 |
| 总训练样本 | ~3520 (80 × 44) | 初始小规模实验 |
| 总测试样本 | ~880 (20 × 44) | 初始小规模实验 |

### 5.2 LoRA 配置（待讨论）

当前 grpo.sh 中的配置：
```yaml
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj]
```

**需要讨论的问题**:
1. **LoRA rank**: 16 是否合适？对于 4B 模型，8/16/32 哪个更好？
2. **Target modules**: 是否需要包含所有 linear layers？还是只 attention？
3. **Alpha**: 32 (2x rank) 是否合适？
4. **Dropout**: 0.05 是否合适？小数据集是否需要更高 dropout？

### 5.3 训练超参数（待讨论）

| 参数 | 建议值 | 说明 |
|------|--------|------|
| 学习率 | 1e-4 ~ 5e-5 | LoRA SFT 常用范围 |
| Batch Size | 4 | per_device |
| Gradient Accumulation | 4 | effective batch = 16 |
| Epochs | 2-3 | 小数据集可能需要更多 epochs |
| Max Length | 4096 | VLM 训练需要 |
| Optimizer | AdamW | 默认 |
| LR Scheduler | cosine | 默认 |

---

## 6. 评估方案

### 6.1 评估指标

| 指标 | 说明 | 适用任务 |
|------|------|---------|
| Exact Match (EM) | 输出与目标完全匹配 | 所有任务 |
| Chemical Validity | 输出是否为有效的化学表示 | OCR, Conversion |
| Chemical Equivalence | 输出与目标是否化学等价（而非字符串匹配） | Conversion |
| Token-level F1 | 输出与目标的 token 级别 F1 | 所有任务 |

### 6.2 评估矩阵

每个实验需要评估：
1. **训练任务**: 看模型是否学会了训练的任务
2. **未训练任务 (zero-shot)**: 看能力是否泛化
3. **跨任务迁移**: 看 OCR 训练是否帮助 conversion，反之亦然

---

## 7. 实施优先级

### Phase 1: 基线实验（优先）
- [ ] E1: LLM-Conversion-Only
- [ ] E2: VLM-OCR-Only
- [ ] E3: VLM-Conversion-Only

### Phase 2: 两阶段实验
- [ ] E4: VLM-OCR→Conversion

### Phase 3: 混合与课程学习
- [ ] E5: VLM-OCR+Conversion-Mixed
- [ ] E6: VLM-OCR→Conversion→Mixed

---

## 8. 预期结果与假设

| 假设 | 验证方式 |
|------|---------|
| H1: VLM 的 OCR 能力可以迁移到 conversion | E2 vs E3 对比 |
| H2: OCR cold start 有助于 conversion | E4 vs E3 对比 |
| H3: 混合训练优于两阶段 | E5 vs E4 对比 |
| H4: 课程学习最优 | E6 vs E4/E5 对比 |
| H5: VLM 的 conversion 能力优于纯 LLM | E3 vs E1 对比 |

---

## 9. 后续扩展

### 9.1 RL/GRPO 阶段
在 SFT 完成后，可以考虑引入 RL 训练：
- **Reward 设计**: 
  - Exact match reward
  - Chemical validity reward (RDKit 验证)
  - Chemical equivalence reward (分子指纹相似度)
- **目标**: 进一步提升 conversion 准确率和鲁棒性

### 9.2 更多任务
- 3D 结构理解
- 性质预测
- 反应预测

---

## 10. 时间线（预估）

| 阶段 | 时间 | 内容 |
|------|------|------|
| Week 1 | 数据准备 + E1/E2/E3 | 基线实验 |
| Week 2 | E4 + 分析 | 两阶段实验 |
| Week 3 | E5/E6 + 分析 | 混合与课程学习 |
| Week 4 | RL 实验（可选） | GRPO 训练 |
