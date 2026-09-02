# Molecule Representation Data Pipeline Plan

## 1. 目标

把 molecule representation 学习任务拆成两层：

1. 先生成“原始 representation 数据”，只负责收集事实；
2. 再基于这些事实生成训练任务（SFT/QA），定义输入、输出和能力目标。

这样做的好处是：

- 数据来源清晰，便于追溯；
- 新 representation 的接入成本低；
- 后续做 ablation、任务混合、比例实验都更方便。

---

## 2. 核心设计：两阶段 Pipeline

### Stage 1: mol-reps-raw

目标：只负责构建每个 molecule 的所有 representation 事实，不关心训练任务。

每个 molecule 对应一条 record，格式示意：

```json
{
  "mol_id": "xxx",
  "canonical": {
    "smiles": "...",
    "selfies": "...",
    "deepsmiles": "...",
    "inchi": "..."
  },
  "random": {
    "smiles": ["...", "...", "..."],
    "selfies": ["...", "...", "..."],
    "deepsmiles": ["...", "...", "..."]
  },
  "iupac": "...",
  "cml": "..."
}
```

原则：

- 只保存事实；
- 不包含 prompt、QA、instruction；
- 作为后续所有任务生成的 source of truth。

### Stage 2: mol-conversion-qa

目标：定义“什么 input → 什么 output → 什么能力”，也就是最终的 SFT 数据。

格式示意：

```json
{
  "instruction": "...",
  "input_rep": "random_smiles",
  "output_rep": "canonical_inchi",
  "task_type": "normalization_cross",
  "input": "...",
  "output": "..."
}
```

这个阶段负责：

- 决定训练任务类型；
- 生成 instruction；
- 组织 input/output 对；
- 形成可直接用于训练的数据。

---

## 3. 任务 taxonomy

### Task 1: Translation

目标：在 canonical representation 之间做双向转换。

#### canonical → canonical

输入：

- canonical SMILES
- canonical SELFIES
- canonical DeepSMILES
- InChI

输出：

- canonical SMILES
- canonical SELFIES
- canonical DeepSMILES
- InChI

任务名：

```text
translation_canonical
```

数量（每个 molecule）：

```text
12
```

---

### Task 2: Understanding

#### 2.1 IUPAC understanding

输入：

- IUPAC

输出：

- canonical SMILES
- canonical SELFIES
- canonical DeepSMILES
- InChI

任务名：

```text
understanding_iupac
```

数量（每个 molecule）：

```text
4
```

#### 2.2 CML understanding

输入：

- CML

输出：

- canonical SMILES
- canonical SELFIES
- canonical DeepSMILES
- InChI

任务名：

```text
understanding_cml
```

数量（每个 molecule）：

```text
4
```

---

### Task 3: Normalization

#### 3.1 Intra normalization

同 representation 类型内部的随机变体归一化到 canonical。

例如：

- random SMILES → canonical SMILES
- random SELFIES → canonical SELFIES
- random DeepSMILES → canonical DeepSMILES

任务名：

```text
normalization_intra
```

数量（每个 molecule，如果随机变体数为 $N$）：

```text
3N
```

#### 3.2 Cross normalization

跨 representation 类型的归一化。

例如：

- random SMILES → canonical SELFIES
- random SMILES → canonical DeepSMILES
- random SMILES → InChI
- random SELFIES → canonical SMILES
- random DeepSMILES → InChI

任务名：

```text
normalization_cross
```

数量（每个 molecule）：

```text
9N
```

---

## 4. 总体数据规模

如果每个 molecule 的随机变体数为 $N$，则每个 molecule 的总任务数为：

```text
20 + 12N
```

其中包括：

- translation: 12
- understanding_iupac: 4
- understanding_cml: 4
- normalization_intra: 3N
- normalization_cross: 9N

---

## 5. 文件布局建议

建议把数据按照“层次化”组织，而不是把所有任务直接混在一起：

```text
data/
├── mol-reps-raw/
│   ├── train.parquet
│   └── metadata.json
│
└── mol-conversion-qa/
    ├── translation.parquet
    ├── understanding.parquet
    ├── normalization.parquet
    └── all.parquet
```

这样后续做 ablation 非常容易，例如：

```python
train = concat([
    translation,
    normalization_intra,
])
```

或者：

```python
train = all - understanding_cml
```

---

## 6. 扩展性设计

这个设计的一个关键优势是：以后新增 representation 时，改动很小。

例如新增：

- SELFIES random variant
- MolJSON
- RDKit MolBlock
- SMARTS
- molecular formula

只需要做两步：

1. 在 Stage 1 中新增字段，例如：

```json
{
  "molblock": "..."
}
```

2. 在 Stage 2 中增加对应的 generator，例如：

```text
generate_understanding_molblock()
```

无需重构整个 pipeline。

---

## 7. 工程落地建议

### 7.1 配置层

当前可以把 representation schema 固定在配置文件中，例如：

- [mol-llm/configs/mol_rep_schema.yaml](configs/mol_rep_schema.yaml)

这份配置用于定义：

- representation 名称；
- display name；
- aliases；
- 类型和 family。

### 7.2 数据生成层

后续实现时，可以分成几个模块：

- `build_raw_reps.py`：生成 Stage 1 数据；
- `generate_qa.py`：生成 Stage 2 任务；
- `mix_datasets.py`：组合不同 task 形成训练集；
- `schema.py`：统一定义字段与 task 类型。

### 7.3 训练层

训练时只消费 Stage 2 的数据，按需要选择子集：

- translation-only
- normalization-only
- all
- ablation variants

---

## 8. 结论

当前这个方案已经不是“单纯造 QA”的思路，而是一个比较标准的数据工程架构：

```text
raw representation layer → task generation layer → training mixture layer
```

这会为后面的研究带来很强的可扩展性：

- 研究 representation 能力；
- 做 task ablation；
- 评估不同训练混合比例；
- 轻松加入新的 molecular representation。
