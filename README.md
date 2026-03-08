# Vector-Memory-Is-All-You-Need: An Intelligent Memory Dialogue System

An advanced dialogue system with vector-centric dynamic clustering memory module, enabling long-term memory management and semantic association reasoning. This system embodies a pragmatic elegance​ through a harmonious balance of simplicity and sophistication. By embracing a vector-centric tri-layer hierarchy, it eliminates unnecessary complexity while delivering robust memory management:

**Pragmatism​** is embedded in its no-loss storage​ (plain-text talk.txt), incremental updates​ (avoiding full reindexing), and pointer-based deduplication​ (storing only tid offsets instead of redundant text). These design choices ensure efficiency, scalability, and minimal resource overhead.

**Elegance​** arises from its dynamic clustering engine, which transforms raw data into semantic hierarchies organically, and its hybrid retrieval pipeline, which synergizes knowledge graphs and vector search for optimal accuracy.

In essence, the architecture achieves elegance through constraint: by limiting itself to three core layers and two retrieval pathways, it avoids over-engineering while solving real-world challenges like long-term memory decay, contextual relevance, and computational efficiency. This philosophy ensures the system remains both intellectually satisfying and operationally effective.

---

## Core Features

1. **Tri-Layer Memory Architecture**
   - **Information Layer**: Pure text storage (`talk.txt`)
   - **Data Layer**: Hierarchical vector database (FAISS index)
   - **Knowledge Layer**: Dynamic clustering semantic network

2. **Intelligent Retrieval Mechanism**
   - Hybrid search: Knowledge graph + vector similarity dual-channel recall
   - Temporal enhancement: Time-weighted semantic scoring
   - Multimodal support (planned: speech/image input)

3. **Self-Evolving Capabilities**
   - Incremental updates: Event-driven partial recomputation
   - Decay mechanism: Time-weighted semantic scoring
   - Cold-start optimization: Rapid clustering for new topics

---

## Quick Start Guide

### Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama service (example)
ollama serve
```

### Run the System

```bash
# Start Gradio interface
python app.py

# Access via web browser:
# http://localhost:7860
```

---

## Configuration Parameters

Key parameters in `config.py`:

```python
# Model Configuration
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama API endpoint
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"  # Embedding model
GENERATION_MODEL = "qwen3:8b"  # Response generation model

# System Parameters
VECTOR_DIM = 768  # Vector dimensionality
MAX_DIALOG_HISTORY = 50  # Maximum conversation history length
CACHE_SIZE = 1000  # Cache capacity for frequent queries
CLUSTER_UPDATE_THRESHOLD = 50  # Trigger clustering after N new entries

# Advanced Parameters
GPU_DEVICE = "0"  # Specify GPU device (e.g., "0" for first GPU)
GPU_LAYERS = -1  # Use all available layers (-1) or specify number
INFERENCE_TIMEOUT = 120  # Inference timeout in seconds
```

---

## Key Modifications Needed

### 1. Ollama Model Prompts
Update the system prompts in `gradio_interface.py`:

```python
# Knowledge retrieval prompt (modify as needed)
system_prompt = """You are an AI assistant with long-term memory capabilities. Follow these rules when answering:

1. Analyze retrieved memory content first
2. Combine retrieved results with the current question
3. Prioritize memory-based answers if relevant
4. Use general knowledge only if memory results are irrelevant
5. Provide natural, coherent answers without listing results directly
6. Properly cite key information from retrieved content

Retrieved memory snippets:
{retrieval_content}

Current user question:
{question}"""
```

### 2. Clustering Parameters
Modify clustering settings in `memory_manager.py`:

```python
# Dynamic clustering parameters
MIN_CLUSTER_SIZE = 5  # Minimum cluster size
MAX_CLUSTERS = 20  # Maximum number of clusters
CLUSTER_UPDATE_THRESHOLD = 50  # Trigger clustering after N new entries
```

---

### 1. Adding New Data Sources
Implement the `DataLoader` interface:
```python
class CustomDataLoader(DataLoaderBase):
    def load(self):
        data = load_custom_data()
        embeddings = embed_data(data)
        store_embeddings(embeddings)
```

### 2. Custom Evaluation Metrics
Define custom metrics using scikit-learn:
```python
from sklearn.metrics import make_scorer

def custom_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

scoring = {'custom_f1': make_scorer(custom_f1_score)}
```

---
