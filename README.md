# BIMS: A Brain-Inspired Memory System for Long-Context Dialogue

**Languages:** [English](README.md) | [中文](README.zh.md)

This repository implements **Brain-Inspired Memory System (BIMS)** from the paper *A Brain-Inspired Memory System for Long-Context Dialogue Agents* (Linrui Xu, 2026). It turns ideas from **Complementary Learning Systems (CLS)** into a computational design: **episodic** vs **semantic** memory, **dual-phase clustering** for consolidation, and **adaptive retrieval** (associative expansion and a **temporal reasoning** mode) to balance **stability** and **plasticity** in very long conversations.

> **Note:** The paper’s experiments use a unified base LLM (e.g. **GPT-OSS-20B**). This codebase defaults to **Ollama** for local embeddings and generation. Reported numbers are from the paper; local runs may differ.

---

## Highlights

- **Problem:** Fixed context windows and weak consolidation hurt long-horizon dialogue—cross-session facts, temporal order, and argumentative coherence.
- **Idea:** Hippocampus–cortex–style split—episodic side stores **time-anchored utterances**; semantic side builds **clustered summaries** of meaning. Two-phase clustering maps **fast encoding** and **slow integration**.
- **Reported results** (test sets, paper Table 2): On **LongMemEval** and **LoCoMo**, BIMS achieves **QA correctness** of **70.7%** and **68.2%**, and **retrieval recall (RR)** of **0.595** and **0.521**. Ablations show **dual-phase clustering** and the **temporal module** matter most for overall and temporal tasks.

---

## Mapping: Paper ↔ This Codebase

| Concept (paper) | Where it lives (code) |
|-----------------|------------------------|
| Episodic memory (paragraph/sentence vectors, timestamps) | `ParagraphNode` / `SentenceNode`, FAISS `VectorStore`, raw text in `data/talk.txt` |
| Semantic memory (cluster centroids + member paragraphs) | `KnowledgeNode`, `knowledge_graph`, `ClusteringLayer` (BIRCH, etc.) on paragraph vectors |
| Dual-phase consolidation: online clustering + slow merge | `add_dialog` → `_update_clusters`; `_hippocampal_consolidation` merges clusters and prunes redundancy |
| Hybrid score: semantics × (semantic weight + recency weight × decay) | `search`, `_recency_weight`, `SEMANTIC_WEIGHT` / `RECENCY_WEIGHT` |
| Implicit associative expansion (no explicit relation graph) | `_associative_retrieval` extends beyond top clusters by centroid similarity |
| Temporal reasoning mode | `is_temporal_task(task_type)` (e.g. `temporal-reasoning`) and `search(..., is_temporal_task=True)`—more paragraph hits, `RECENT_TEMPORAL_TURNS`, time-ordered sorting |
| Summary semantic nodes | `SummaryNode`, `_update_summary_memory`, `_summary_search` |
| Ablation switches | `VectorMemoryManager(..., ablation={...})`: `no_temporal`, `no_assoc`, `single_stage_cluster`, `balanced_sem_rec_weights` |

Engineering extras: **FAISS IVFPQ** when large enough, **LRU** cache for embeddings and search, cluster metrics and `report_retrieval_success`.

---

## Repository layout

```
├── app.py                 # Gradio web UI
├── memory_manager.py      # BIMS core: storage, clustering, retrieval
├── config.py              # Paths and Ollama model names
├── ablation_eval.py       # LongMemEval subset ablations
└── eval/
    ├── eval_new.py        # LongMemEval evaluation (Ollama)
    └── quick_eval.py      # Lightweight smoke tests
```

Data and indexes default to `data/talk.txt`, `data/vectors/`, `data/knowledge/` (see `config.py`).

---

## Requirements

- **Python 3.10+** (recommended)
- **Ollama** with embedding and chat models matching `config.py` (or your own names)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

GPU users may replace `faiss-cpu` with a suitable `faiss-gpu` build for their CUDA version; the code imports `faiss` the same way.

---

## Quick start

```bash
ollama serve

python app.py
# Open http://localhost:7860
```

Main knobs: **`config.py`** — `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`, `GENERATION_MODEL`, `VECTOR_DIM`, etc.

---

## Key hyperparameters (paper Table 1; see `memory_manager.py`)

| Parameter | Role | Default (code) |
|-----------|------|----------------|
| `MIN_CLUSTER_SIZE` | Min paragraphs for a stable cluster | 3 |
| `MERGE_SIMILARITY_THRESHOLD` | Merge threshold θ_M in slow consolidation | 0.2 |
| `REDUNDANT_SIMILARITY_THRESHOLD` | Drop near-duplicate vectors | 0.85 |
| `SEMANTIC_WEIGHT` / `RECENCY_WEIGHT` | α vs recency in retrieval | 0.7 / 0.3 |
| `CLUSTER_UPDATE_THRESHOLD` | New entries before a clustering refresh | 4 |
| `SUMMARY_UPDATE_THRESHOLD` | New paragraphs before a new summary node | 20 |
| `ASSOCIATIVE_EXPAND_CLUSTERS` | Clusters to explore in associative expansion | 3 |
| `MAX_PARAS_PER_CLUSTER` | Max paragraphs returned per cluster | 3 |
| `RECENT_TEMPORAL_TURNS` | Recent turns to add in temporal mode (∆t window) | 6 |
| `MIN_SUMMARY_SIMILARITY`, etc. | Summary/ knowledge thresholds | Overridable via `eval_config.json` |

`CLUSTER_UPDATE_THRESHOLD` also appears in `config.py`; **the constants at the top of `memory_manager.py` are authoritative** for tuning aligned with the paper.

---

## Evaluation & ablations

- **LongMemEval** (from repo root):  
  `python eval/eval_new.py --dataset oracle --sample_size 500 --config eval_config.json`  
  Prepare `eval_config.json` and dataset paths (`benchmark_path`, etc. in the script defaults).
- **Ablations:**  
  `python ablation_eval.py --sample_size 100 --output_dir results/ablation`  
  Compares full BIMS vs. variants without temporal handling, associative retrieval, dual-phase clustering, or semantic-weight emphasis (paper Table 4).

Ensure your **judge / generation** model is available in Ollama before running eval.

---

## Citation

Please cite the paper if you use this idea or codebase. Correspondence: `231224006@cupl.edu.cn`.

---

## Disclaimer

Empirical claims, limitations, and future work follow **Sections 5–7** of the paper. This code is provided as-is for research and reproduction.
