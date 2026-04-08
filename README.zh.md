# BIMS：面向长上下文对话的脑启发记忆系统

**语言：** [English](README.md) | [中文](README.zh.md)

本仓库对应论文 **「A Brain-Inspired Memory System for Long-Context Dialogue Agents」**（Linrui Xu, 2026）的开源实现：**Brain-Inspired Memory System（BIMS）**。论文将认知科学中的 **互补学习系统（Complementary Learning Systems, CLS）** 思想计算化：用 **情景记忆** 与 **语义记忆** 分离、**双阶段聚类巩固**，以及 **可适配备检索**（含关联扩展与 **时序推理模式**），在超长对话中平衡记忆的 **稳定性** 与 **可塑性**。

> **说明**：论文实验使用 **GPT-OSS-20B** 等统一基座；本仓库默认通过 **Ollama** 调用本地嵌入与生成模型，便于复现与部署，数值结果以论文为准。

---

## 论文要点

- **问题**：固定上下文窗口与缺乏结构化巩固机制，使长对话系统在跨会话事实整合、时序与论证性推理上易退化。
- **思路**：模仿海马—皮层分工——情景侧保留带时间锚的 **话语片段**，语义侧通过聚类形成 **概括性知识簇**；两阶段聚类对应 **快速编码** 与 **慢速整合**。
- **主要结果**（测试集，论文 Table 2）：在 **LongMemEval** 与 **LoCoMo** 上，BIMS 的 **问答正确率（QA）** 分别为 **70.7%** 与 **68.2%**；**检索召回（RR）** 分别为 **0.595** 与 **0.521**。消融实验表明 **双阶段聚类** 与 **时序模块** 对整体与时序类任务尤为关键。

---

## 与本仓库代码的对应关系

| 论文概念 | 实现位置（概要） |
|---------|------------------|
| 情景记忆（段落/句子向量、时间戳） | `ParagraphNode` / `SentenceNode`，向量入 FAISS `VectorStore`，原文追加写入 `data/talk.txt` |
| 语义记忆（簇质心 + 成员段落） | `KnowledgeNode` 与 `knowledge_graph`，由 `ClusteringLayer`（BIRCH 等）在段落向量上聚类得到 |
| 双阶段巩固：阶段一在线聚类；阶段二慢合并 | `add_dialog` 触发 `_update_clusters`；`_hippocampal_consolidation` 执行簇合并与冗余清理（`merge_similar_clusters` 等） |
| 混合检索分数：语义 ×（语义权重 + 新近度权重 × 衰减） | `search` 中 `_recency_weight` 与 `SEMANTIC_WEIGHT` / `RECENCY_WEIGHT` |
| 隐式关联扩展（无显式关系图） | `_associative_retrieval`：在 Top 簇基础上按质心相似度扩展 |
| 时序推理模式 | `is_temporal_task(task_type)`（如 `temporal-reasoning`）与 `search(..., is_temporal_task=True)`：提高段落检索量、补充最近 `RECENT_TEMPORAL_TURNS` 轮并按时间排序 |
| 摘要型语义节点 | `SummaryNode`，周期性 `_update_summary_memory`，`_summary_search` 参与检索 |
| 消融开关 | `VectorMemoryManager(..., ablation={...})`：`no_temporal` / `no_assoc` / `single_stage_cluster` / `balanced_sem_rec_weights` |

工程特性还包括：向量规模足够时 **FAISS IVFPQ** 压缩、**LRU** 缓存嵌入与检索结果、簇指标与 `report_retrieval_success` 统计等。

---

## 目录结构

```
├── app.py                 # Gradio Web 对话界面
├── memory_manager.py      # BIMS 核心：存储、聚类、检索
├── config.py              # 路径与 Ollama 模型名等
├── ablation_eval.py       # LongMemEval 子集消融脚本
└── eval/
    ├── eval_new.py        # LongMemEval 评测主流程（Ollama）
    └── quick_eval.py      # 轻量快速试验
```

数据与索引默认写入 `data/talk.txt`、`data/vectors/`、`data/knowledge/`（见 `config.py`）。

---

## 环境依赖

需已安装 **Python 3.10+**（推荐）、**Ollama**，并拉取 `config.py` 中配置的嵌入与对话模型（可按需改名）。

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

若使用 GPU，可将 `faiss-cpu` 换为对应 CUDA 版本的 **faiss-gpu**；代码对 `faiss` 的导入方式不变。

---

## 快速开始

```bash
# 启动 Ollama（示例）
ollama serve

# 在仓库根目录启动 Web 界面
python app.py
# 浏览器访问 http://localhost:7860
```

主要配置见 **`config.py`**：`OLLAMA_BASE_URL`、`EMBEDDING_MODEL`、`GENERATION_MODEL`、`VECTOR_DIM` 等。

---

## 核心超参数（与论文 Table 1 对齐，见 `memory_manager.py`）

| 参数 | 含义 | 默认值（代码） |
|------|------|----------------|
| `MIN_CLUSTER_SIZE` | 稳定簇的最小段落数 | 3 |
| `MERGE_SIMILARITY_THRESHOLD` | 慢巩固时簇合并相似度阈值 θ_M | 0.2 |
| `REDUNDANT_SIMILARITY_THRESHOLD` | 冗余向量剔除阈值 | 0.85 |
| `SEMANTIC_WEIGHT` / `RECENCY_WEIGHT` | 检索中语义 vs 新近度 α | 0.7 / 0.3 |
| `CLUSTER_UPDATE_THRESHOLD` | 新增多少条向量后触发聚类更新 | 4 |
| `SUMMARY_UPDATE_THRESHOLD` | 多少新段落后更新摘要节点 | 20 |
| `ASSOCIATIVE_EXPAND_CLUSTERS` | 关联扩展考虑的簇数 | 3 |
| `MAX_PARAS_PER_CLUSTER` | 单簇最多返回段落数 | 3 |
| `RECENT_TEMPORAL_TURNS` | 时序模式下补充的近期轮数（∆t 窗口） | 6 |
| `MIN_SUMMARY_SIMILARITY` 等 | 摘要/知识检索阈值 | 可由 `eval_config.json` 覆盖 |

`config.py` 中的 `CLUSTER_UPDATE_THRESHOLD` 会与 `memory_manager` 内部分常量并存；**以 `memory_manager.py` 顶部常量为准** 或与论文一致的调参说明。

---

## 评测与消融

- **LongMemEval**：在仓库根目录执行，例如  
  `python eval/eval_new.py --dataset oracle --sample_size 500 --config eval_config.json`  
  需准备 `eval_config.json` 与数据路径（见评测脚本内 `benchmark_path` 等默认项）。
- **消融**：  
  `python ablation_eval.py --sample_size 100 --output_dir results/ablation`  
  对比完整模型与去掉时序、关联、双阶段聚类、语义权重等变体（与论文 Table 4 对应）。

评测依赖 Ollama 上的 **评判/生成模型**；首次运行前请确认 `eval_config.json` 与数据集路径。

---

## 引用

若使用本仓库或论文思想，请引用论文原文。通讯作者：`231224006@cupl.edu.cn`。

---

## 许可与声明

实验结果、局限性与未来方向以论文 **第 5–7 节** 为准；本代码按「原样」提供，用于研究与复现。
