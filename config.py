import os
from datetime import datetime

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TALK_FILE = os.path.join(BASE_DIR, "data/talk.txt")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "data/vectors")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data/knowledge")

# 创建目录
os.makedirs(os.path.dirname(TALK_FILE), exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# Ollama配置
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
GENERATION_MODEL = "qwen3:8b"

# GPU配置
CUDA_DEVICE = "0"  # 指定使用GPU 0
GPU_LAYERS = -1    # -1表示使用所有可用层，或指定具体层数如 20

# 向量配置
VECTOR_DIM = 768
TOP_K_RETRIEVAL = 3

# 记忆配置
MAX_DIALOG_HISTORY = 50
CACHE_SIZE = 1000
CLUSTER_UPDATE_THRESHOLD = 5

# 推理配置
INFERENCE_TIMEOUT = 120  # 增加超时时间以适应GPU推理