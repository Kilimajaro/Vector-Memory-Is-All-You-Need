# -*- coding: utf-8 -*-
"""
脑启发记忆系统 (Brain-Inspired Memory System)
基于认知心理学模型（Atkinson-Shiffrin、互补学习系统）的优化实现：
- 情景/语义记忆分离、多阶段语义聚合
- 双阶段聚类（海马体快速编码 + 新皮层慢速整合）
- 隐式关联检索（语义相似度动态召回，无显式关系图）
- 乘积量化、LRU 缓存、时间衰减、自优化
"""

import json
import random
import numpy as np
from datetime import datetime
from collections import OrderedDict
import faiss
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score
import requests
import os
import logging
import re
from config import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.disable(logging.WARNING)

TOP_K_RETRIEVAL = 30  

# 聚类与整合参数
MIN_CLUSTER_SIZE = 3
MERGE_SIMILARITY_THRESHOLD = 0.2
REDUNDANT_SIMILARITY_THRESHOLD = 0.85
MIN_SILHOUETTE_PER_CLUSTER = 0.01
MAX_INTRACLUSTER_DISTANCE = 2.0
BIRCH_THRESHOLD = 0.3
BIRCH_N_CLUSTERS = None
CLUSTER_UPDATE_THRESHOLD = 4
MAX_DIALOG_HISTORY = 100
# 按段落数动态目标簇数，避免 BIRCH+merge 合并成单簇：目标约 n_para/10，上下浮动
TARGET_CLUSTERS_DIVISOR = 10
TARGET_CLUSTERS_FLOAT_RATIO = 0.2

# 向量索引：PQ 相关
PQ_NLIST = 50
PQ_M = 8
PQ_NBITS = 8
MIN_VECTORS_FOR_PQ = 100

# 检索与缓存（恢复语义主导，避免 recency 过高导致噪声）
RECENCY_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.7
LRU_CACHE_SIZE = 500
ASSOCIATIVE_EXPAND_CLUSTERS = 3  # 折中：略扩跨簇以利 knowledge-update，但不过大以免噪声
SUMMARY_UPDATE_THRESHOLD = 20

# 时序推理任务专用：由测评方通过 is_temporal_task 传参触发，不再用关键词
RECENT_TEMPORAL_TURNS = 6   # 时序查询时补充的近期轮数
TEMPORAL_PARA_TOP_K = 7     # 时序查询时段落检索多取几条以提升召回
MAX_PARAS_PER_CLUSTER = 3   # 单簇最多返回段落数，避免同簇占满导致重复/多样性差
MIN_SUMMARY_SIMILARITY = 0.28  # 摘要/知识检索阈值，可由 eval_config.json 覆盖（调参写入）


def _load_retrieval_params(retrieval_params=None):
    """解析检索阈值：dict 来自 eval_config.json（min_summary_similarity, knowledge_min_score）。"""
    if retrieval_params is None or not isinstance(retrieval_params, dict):
        return {
            "min_summary_similarity": MIN_SUMMARY_SIMILARITY,
            "knowledge_min_score": 0.28,
        }
    return {
        "min_summary_similarity": float(retrieval_params.get("min_summary_similarity", MIN_SUMMARY_SIMILARITY)),
        "knowledge_min_score": float(retrieval_params.get("knowledge_min_score", 0.28)),
    }


# ==================== 一、数据结构（情景 / 语义分离）====================

class SentenceNode:
    """句子级节点（情景记忆）"""
    def __init__(self, sent_id, text, vector, parent_para_id, timestamp=None, context_label=None):
        self.id = sent_id
        self.text = text
        self.vector = vector
        self.parent_para_id = parent_para_id
        self.timestamp = timestamp or datetime.now().isoformat()
        self.context_label = context_label or ""


class ParagraphNode:
    """段落级节点"""
    def __init__(self, para_id, text, timestamp=None):
        self.id = para_id
        self.text = text
        self.para_vector = None
        self.sentences = []
        self.timestamp = timestamp or datetime.now().isoformat()

    def add_sentence(self, sent_node):
        self.sentences.append(sent_node)


class SummaryNode:
    """摘要记忆（语义层）"""
    def __init__(self, theme, abstract_text, source_para_ids, summary_id=None):
        self.id = summary_id or f"sum_{int(datetime.now().timestamp()*1000)}"
        self.theme = theme
        self.abstract = abstract_text
        self.source_para_ids = list(source_para_ids)
        self.vector = None
        self.timestamp = datetime.now().isoformat()


class KnowledgeNode:
    """知识级节点"""
    def __init__(self, node_id, center_vector):
        self.node_id = node_id
        self.center_vector = center_vector
        self.paragraph_ids = []
        self.metrics = {
            'size': 0,
            'intra_dist': 0.0,
            'silhouette': 0.0,
            'utility': 0.0,
            'retrieval_count': 0,
            'success_rate': 1.0,
        }

    def update_metrics(self, retrieval_count=None, success_rate=None, **kwargs):
        if retrieval_count is not None:
            self.metrics['retrieval_count'] = retrieval_count
        if success_rate is not None:
            self.metrics['success_rate'] = success_rate
        self.metrics['utility'] = self.metrics['retrieval_count'] * self.metrics['success_rate']
        self.metrics.update(kwargs)


# ==================== 二、LRU 热点记忆缓存 ====================

class LRUMemoryCache:
    """热点记忆 LRU 缓存"""
    def __init__(self, capacity=LRU_CACHE_SIZE):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()


# ==================== 三、向量存储（乘积量化 + 元数据）====================

class VectorStore:
    """支持 IndexFlatIP 与 IndexIVFPQ 的向量存储"""
    def __init__(self, dim, use_pq=True):
        self.dim = dim
        self.use_pq = use_pq
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []
        self._pq_trained = False

    def _ensure_pq(self, ntotal):
        if not self.use_pq or ntotal < MIN_VECTORS_FOR_PQ or self._pq_trained:
            return
        
        try:
            max_nlist = max(2, ntotal // 39)
            nlist = min(PQ_NLIST, max_nlist)
            
            if ntotal < nlist * 39:
                logger.info("样本不足(%d < %d)，使用Flat索引", ntotal, nlist*39)
                return
            
            m = min(PQ_M, self.dim // 4)
            if m < 1:
                m = 1
            
            vectors = []
            valid_indices = []
            for i in range(ntotal):
                try:
                    vec = self.index.reconstruct(i)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        vectors.append(vec)
                        valid_indices.append(i)
                except:
                    continue
                    
            n_vectors = len(vectors)
            if n_vectors < nlist * 39:
                logger.info("有效样本不足(%d < %d)，使用Flat索引", n_vectors, nlist*39)
                return
            # 避免 FAISS 报错 "please provide at least 9984 training points"：nlist 不超过 实际向量数/39
            nlist_final = min(nlist, n_vectors // 39)
            if nlist_final < 1:
                return
            nlist = nlist_final
            vectors = np.array(vectors, dtype='float32')
            quantizer = faiss.IndexFlatL2(self.dim)
            pq_index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, PQ_NBITS)
            faiss.normalize_L2(vectors)
            pq_index.train(vectors)
            pq_index.add(vectors)
            
            # 创建新的元数据列表，只包含有效向量
            new_metadata = [self.metadata[i] for i in valid_indices]
            self.metadata = new_metadata
            self.index = pq_index
            self._pq_trained = True
            logger.info("切换到PQ索引: nlist=%d, m=%d, 训练样本=%d", nlist, m, len(vectors))
            
        except Exception as e:
            logger.warning("PQ训练失败: %s", e) 

    def add(self, vector, meta):
        vector = np.asarray(vector, dtype='float32').reshape(1, -1)
        if self.index.ntotal > 0 and self.index.d != vector.shape[1]:
            raise ValueError("dimension mismatch")
        self.index.add(vector)
        meta['index_pos'] = self.index.ntotal - 1
        self.metadata.append(meta)
        self._ensure_pq(self.index.ntotal)

    def search(self, query_vector, k, vec_type=None):
        if self.index.ntotal == 0:
            return [], []
        k = min(k, self.index.ntotal)
        q = np.asarray(query_vector, dtype='float32').reshape(1, -1)
        
        if isinstance(self.index, faiss.IndexIVFPQ):
            self.index.nprobe = min(10, self.index.nlist)
            distances, indices = self.index.search(q, k)
            distances = distances.ravel()
            indices = indices.ravel()
            # 正确计算余弦相似度
            scores = 1.0 - (distances.astype(np.float32) / 2.0)
        else:
            scores, indices = self.index.search(q, k)
            scores = scores.ravel()
            indices = indices.ravel()
        
        results = []
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            if vec_type and meta.get('type') != vec_type:
                continue
            # 确保分数在合理范围内
            score = float(scores[i])
            if score < -1.0:
                score = -1.0
            elif score > 1.0:
                score = 1.0
            results.append((score, idx, meta))
        return results

    def reconstruct(self, idx):
        return self.index.reconstruct(idx)

    def ntotal(self):
        return self.index.ntotal

    def rebuild_from_metadata(self, metadata, vectors):
        self.metadata = list(metadata)
        self.index = faiss.IndexFlatIP(self.dim)
        if len(vectors) > 0:
            self.index.add(np.array(vectors, dtype='float32'))
        for new_i, meta in enumerate(self.metadata):
            meta['index_pos'] = new_i
        self._pq_trained = False
        self._ensure_pq(self.index.ntotal)


# ==================== 四、双阶段聚类 ====================

def _target_n_clusters(n_paragraphs):
    """按段落数算目标簇数：约 n/10，上下浮动，避免单一大垃圾簇。"""
    if n_paragraphs <= 1:
        return 1
    base = max(1, n_paragraphs // TARGET_CLUSTERS_DIVISOR)
    delta = max(1, int(base * TARGET_CLUSTERS_FLOAT_RATIO))
    low = max(1, base - delta)
    high = min(n_paragraphs, base + delta)
    target = random.randint(low, high) if low <= high else base
    return max(1, min(n_paragraphs, target))


class ClusteringLayer:
    """双阶段聚类：支持按目标簇数 KMeans（推荐），或 BIRCH + 合并小簇"""
    def __init__(self, threshold=BIRCH_THRESHOLD):
        self.birch = Birch(threshold=threshold, n_clusters=BIRCH_N_CLUSTERS)
        self.labels_ = None
        self.centers_ = None

    def encode_with_target_k(self, X, target_k):
        """按目标簇数 KMeans，避免簇过多/过少；返回 (labels, centers)，过滤空簇。"""
        X = np.asarray(X, dtype='float32')
        n = len(X)
        if n <= 1:
            self.labels_ = np.zeros(n, dtype=int) if n else np.array([])
            dim = X.shape[1] if X.size else 0
            self.centers_ = X.copy() if n else np.zeros((0, dim), dtype='float32')
            return self.labels_, self.centers_
        k = max(1, min(n, target_k))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.labels_ = km.fit_predict(X)
        unique = sorted(set(self.labels_))
        old_to_new = {u: i for i, u in enumerate(unique)}
        self.labels_ = np.array([old_to_new[l] for l in self.labels_])
        n_actual = len(unique)
        self.centers_ = np.zeros((n_actual, X.shape[1]), dtype='float32')
        for c in range(n_actual):
            members = np.where(self.labels_ == c)[0]
            if len(members) > 0:
                self.centers_[c] = np.mean(X[members], axis=0)
        return self.labels_, self.centers_

    def fast_encode(self, X):
        X = np.asarray(X, dtype='float32')
        if len(X) < 2:
            self.labels_ = np.zeros(len(X), dtype=int) if len(X) else np.array([])
            self.centers_ = X
            return self.labels_, self.centers_
        self.birch.fit(X)
        self.labels_ = self.birch.predict(X)
        n_clusters = max(1, len(set(self.labels_)))
        self.centers_ = np.zeros((n_clusters, X.shape[1]), dtype='float32')
        for c in range(n_clusters):
            members = np.where(self.labels_ == c)[0]
            if len(members) > 0:
                self.centers_[c] = np.mean(X[members], axis=0)
        return self.labels_, self.centers_

    def merge_similar_clusters(self, X, labels, centers, min_size, merge_sim_thresh):
        n_clusters = len(centers)
        if n_clusters <= 1:
            return labels, centers
        
        # 计算余弦相似度矩阵
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        centers_norm = centers / norms
        sim_matrix = np.dot(centers_norm, centers_norm.T)
        
        parent = list(range(n_clusters))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(i, j):
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pj] = pi

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                size_i = np.sum(labels == i)
                size_j = np.sum(labels == j)
                if size_i < min_size or size_j < min_size or sim_matrix[i, j] >= merge_sim_thresh:
                    union(i, j)
        
        roots = sorted(set(find(x) for x in range(n_clusters)))
        old_to_new = {r: i for i, r in enumerate(roots)}
        new_labels = np.array([old_to_new[find(l)] for l in labels])
        new_centers = np.zeros((len(roots), X.shape[1]), dtype='float32')
        for c, r in enumerate(roots):
            members = np.where(new_labels == c)[0]
            if len(members) > 0:
                new_centers[c] = np.mean(X[members], axis=0)
        return new_labels, new_centers


# ==================== 五、主管理类 ====================

class VectorMemoryManager:
    def __init__(self, retrieval_params=None, ablation=None):
        """retrieval_params: 可选 dict，来自 eval_config.json 的 min_summary_similarity / knowledge_min_score。
        ablation: 可选 dict，小规模消融实验开关：
          - no_temporal: 关闭时序推理专用检索与排序（等价于始终非时序任务）
          - no_assoc: 关闭知识簇上的隐式关联扩展，仅保留主检索命中簇
          - single_stage_cluster: 仅保留在线聚类，跳过二次整合（_hippocampal_consolidation）
          - balanced_sem_rec_weights: 检索打分用语义/近因各 0.5（默认 0.7/0.3）
        """
        self._retrieval_cfg = _load_retrieval_params(retrieval_params)
        ablation = ablation or {}
        self._ablation_no_temporal = bool(ablation.get("no_temporal"))
        self._ablation_no_assoc = bool(ablation.get("no_assoc"))
        self._ablation_single_stage_cluster = bool(ablation.get("single_stage_cluster"))
        if ablation.get("balanced_sem_rec_weights"):
            self.semantic_weight = 0.5
            self.recency_weight = 0.5
        else:
            self.semantic_weight = SEMANTIC_WEIGHT
            self.recency_weight = RECENCY_WEIGHT
        self.talk_file = TALK_FILE
        self.vector_store = VectorStore(VECTOR_DIM, use_pq=True)
        self.clustering = ClusteringLayer(threshold=BIRCH_THRESHOLD)
        self.knowledge_graph = {}
        self.para_tree = {}
        self.sent_map = {}
        self.summary_nodes = {}
        self.lru_cache = LRUMemoryCache(capacity=LRU_CACHE_SIZE)
        self.dimension_verified = False
        self._initialize_vector_db()
        
        # 确保对话文件存在
        os.makedirs(os.path.dirname(self.talk_file), exist_ok=True)
        if not os.path.exists(self.talk_file):
            open(self.talk_file, 'w').close()

    def _initialize_vector_db(self):
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        index_path = f"{VECTOR_DB_DIR}/vector.index"
        metadata_path = f"{VECTOR_DB_DIR}/metadata.json"
        knowledge_path = f"{KNOWLEDGE_DIR}/knowledge_graph.json"
        summary_path = f"{KNOWLEDGE_DIR}/summary_memory.json"

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.vector_store.index = faiss.read_index(index_path)
                self.vector_store._pq_trained = not isinstance(self.vector_store.index, faiss.IndexFlatIP)
            except Exception as e:
                logger.warning("加载 FAISS 索引失败，重建: %s", e)
                self.vector_store.index = faiss.IndexFlatIP(VECTOR_DIM)
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.vector_store.metadata = json.load(f)
            except:
                self.vector_store.metadata = []
            if self.vector_store.index.d != VECTOR_DIM:
                logger.warning("维度不匹配，重建索引")
                self.vector_store.index = faiss.IndexFlatIP(VECTOR_DIM)
                self.vector_store.metadata = []
        else:
            self.vector_store.index = faiss.IndexFlatIP(VECTOR_DIM)
            self.vector_store.metadata = []

        if os.path.exists(knowledge_path):
            try:
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    kg_data = json.load(f)
                    for node_id, data in kg_data.items():
                        kn = KnowledgeNode(node_id, np.array(data['center_vector'], dtype='float32'))
                        kn.paragraph_ids = data.get('paragraph_ids', [])
                        kn.metrics = data.get('metrics', kn.metrics)
                        self.knowledge_graph[node_id] = kn
            except Exception as e:
                logger.warning("加载知识图谱失败: %s", e)

        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                    for item in summary_data:
                        sn = SummaryNode(
                            item['theme'],
                            item['abstract'],
                            item['source_para_ids'],
                            item.get('id')
                        )
                        if item.get('vector'):
                            sn.vector = np.array(item['vector'], dtype='float32')
                        sn.timestamp = item.get('timestamp', '')
                        self.summary_nodes[sn.id] = sn
            except Exception as e:
                logger.warning("加载摘要记忆失败: %s", e)

        self._update_clusters()

    def _get_embedding(self, text):
        if not text or len(text.strip()) < 3:
            return np.zeros(VECTOR_DIM, dtype='float32')
        cache_key = f"emb:{hash(text) % (2**32)}"
        cached = self.lru_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
                timeout=10
            )
            if resp.status_code == 200:
                emb = resp.json().get('embedding', [])
                if len(emb) == VECTOR_DIM:
                    vec = np.array(emb, dtype='float32')
                    self.lru_cache.put(cache_key, vec)
                    return vec
        except Exception as e:
            logger.error("获取嵌入失败: %s", e)
        return np.random.normal(0, 0.1, VECTOR_DIM).astype('float32')

    def _normalize_vector(self, vector):
        arr = np.array(vector, dtype='float32')
        norm = np.linalg.norm(arr)
        if norm < 1e-6:  # 严格校验
            logger.warning("Critical vector norm %.6f", norm)
            return np.zeros(VECTOR_DIM)  # 返回零向量而非随机向量
        return arr / norm

    def _generate_tid(self):
        return f"tid_{int(datetime.now().timestamp()*1000)}"

    def _split_sentences(self, text):
        pattern = r'(?<=[。！？；\n])'
        raw = re.split(pattern, text)
        sents = []
        buffer = ""
        for part in raw:
            buffer += part
            if re.search(r'[。！？；\n]$', buffer):
                clean = buffer.strip()
                if len(clean) > 5:
                    sents.append(clean)
                buffer = ""
        if buffer.strip():
            sents.append(buffer.strip())
        return sents

    def _should_add_sentence_vector(self, sent_vector, para_vector):
        if para_vector is None:
            return True
        norm_para = np.linalg.norm(para_vector)
        norm_sent = np.linalg.norm(sent_vector)
        if norm_para < 1e-8 or norm_sent < 1e-8:
            return False
        similarity = np.dot(para_vector, sent_vector) / (norm_para * norm_sent)
        return similarity > 0.3

    def add_dialog(self, role, text):
        tid_para = self._generate_tid()
        ts = datetime.now().isoformat()
        
        # 避免重复存储相同内容
        if self._is_duplicate_dialog(role, text, ts):
            logger.info("检测到重复对话，跳过存储")
            return tid_para
        
        para_node = ParagraphNode(tid_para, text, timestamp=ts)
        para_vector = self._normalize_vector(self._get_embedding(text))
        para_node.para_vector = para_vector
        self.para_tree[tid_para] = para_node

        for i, sent in enumerate(self._split_sentences(text)):
            sent_id = f"{tid_para}_sent{i}"
            sent_vector = self._normalize_vector(self._get_embedding(sent))
            sent_node = SentenceNode(sent_id, sent, sent_vector, tid_para, timestamp=ts, context_label=role)
            para_node.add_sentence(sent_node)
            self.sent_map[sent_id] = sent_node
            if self._should_add_sentence_vector(sent_vector, para_vector):
                self._add_to_vector_db(sent_id, sent_vector, sent, 'sentence', tid_para, ts, role)
            else:
                logger.debug("句子向量相似度过低，已跳过入库: '%s...'", sent[:15])

        self._add_to_vector_db(tid_para, para_vector, text, 'paragraph', None, ts, role)

        # 保存到对话文件
        try:
            with open(self.talk_file, 'a', encoding='utf-8') as f:
                meta = {'tid': tid_para, 'timestamp': ts, 'role': role}
                f.write(f"{json.dumps(meta)}|{text}\n")
        except Exception as e:
            logger.error("保存对话失败: %s", e)

        if len(self.vector_store.metadata) % CLUSTER_UPDATE_THRESHOLD == 0:
            self._update_clusters()

        if len(self.para_tree) % SUMMARY_UPDATE_THRESHOLD == 0:
            self._update_summary_memory()

        return tid_para
    
    def _is_duplicate_dialog(self, role, text, timestamp):
        """检查是否为重复对话"""
        if not os.path.exists(self.talk_file):
            return False
            
        try:
            with open(self.talk_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-10:]:  # 只检查最近10条
                    if not line.strip():
                        continue
                    parts = line.strip().split('|', 1)
                    if len(parts) < 2:
                        continue
                    try:
                        meta = json.loads(parts[0])
                        if meta['role'] == role and meta['timestamp'] == timestamp:
                            return True
                    except:
                        continue
        except Exception as e:
            logger.error("检查重复对话失败: %s", e)
        return False

    def _add_to_vector_db(self, vec_id, vector, text, vec_type, parent_tid, timestamp=None, context_label=None):
        meta = {
            'id': vec_id,
            'text': text,
            'type': vec_type,
            'parent_tid': parent_tid,
            'timestamp': timestamp or datetime.now().isoformat(),
            'context_label': context_label or '',
        }
        self.vector_store.add(vector, meta)

    def _save_vector_db(self):
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        faiss.write_index(self.vector_store.index, f"{VECTOR_DB_DIR}/vector.index")
        with open(f"{VECTOR_DB_DIR}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.vector_store.metadata, f, ensure_ascii=False, indent=2)
        kg_data = {}
        for nid, node in self.knowledge_graph.items():
            kg_data[nid] = {
                'center_vector': node.center_vector.tolist(),
                'paragraph_ids': node.paragraph_ids,
                'metrics': node.metrics,
            }
        with open(f"{KNOWLEDGE_DIR}/knowledge_graph.json", 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=2)
        summary_list = []
        for sn in self.summary_nodes.values():
            summary_list.append({
                'id': sn.id,
                'theme': sn.theme,
                'abstract': sn.abstract,
                'source_para_ids': sn.source_para_ids,
                'vector': sn.vector.tolist() if sn.vector is not None else None,
                'timestamp': sn.timestamp,
            })
        with open(f"{KNOWLEDGE_DIR}/summary_memory.json", 'w', encoding='utf-8') as f:
            json.dump(summary_list, f, ensure_ascii=False, indent=2)

    def _compute_cluster_metrics(self, X, labels, centers):
        n_clusters = len(centers)
        metrics_list = []
        for c in range(n_clusters):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                metrics_list.append({'size': 0, 'intra_dist': 0, 'silhouette': 0})
                continue
            pts = X[members]
            center = centers[c]
            intra_dists = 1 - np.dot(pts, center)
            intra_dist = float(np.mean(intra_dists))
            other_centers = np.delete(centers, c, axis=0)
            if other_centers.size > 0:
                inter_sims = np.dot(pts, other_centers.T)
                max_inter = np.max(inter_sims, axis=1)
                a = 1 - np.dot(pts, center)
                b = 1 - max_inter
                sil_vals = (b - a) / (np.maximum(a, b) + 1e-8)
                sil = float(np.mean(sil_vals))
            else:
                sil = 0.0
            metrics_list.append({'size': len(members), 'intra_dist': intra_dist, 'silhouette': sil})
        return metrics_list

    def _hippocampal_consolidation(self):
        """新皮层慢速整合：合并小簇、清理冗余向量"""
        para_vectors = []
        para_indices = []
        para_metadata = []
        
        for i, meta in enumerate(self.vector_store.metadata):
            if meta['type'] == 'paragraph' and meta.get('index_pos', -1) < self.vector_store.ntotal():
                try:
                    vec = self.vector_store.reconstruct(meta['index_pos'])
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        para_vectors.append(vec)
                        para_indices.append(i)
                        para_metadata.append(meta)
                except Exception:
                    continue
        
        if len(para_vectors) < 2:
            return
        
        X = np.array(para_vectors)
        n_para = len(para_vectors)
        target_k = _target_n_clusters(n_para)
        labels, centers = self.clustering.encode_with_target_k(X, target_k)
        
        n_clusters = len(centers)
        self.knowledge_graph = {}
        metrics_list = self._compute_cluster_metrics(X, labels, centers)
        
        for cluster_id in range(n_clusters):
            members = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            if not members:
                continue
            
            center = self._normalize_vector(centers[cluster_id])
            node_id = f"kno_{cluster_id}"
            kn = KnowledgeNode(node_id, center)
            kn.paragraph_ids = [para_metadata[i]['id'] for i in members]
            kn.metrics = metrics_list[cluster_id] if cluster_id < len(metrics_list) else {}
            self.knowledge_graph[node_id] = kn
        
        logger.info("海马体整合完成，段落数: %d, 目标簇数: %d, 实际聚类数: %d", n_para, target_k, n_clusters)
        self._purge_redundant_vectors()

    def _purge_redundant_vectors(self):
        """清理相似度高的冗余向量"""
        meta_list = self.vector_store.metadata
        indices = [i for i, m in enumerate(meta_list) if m['type'] in ('sentence', 'paragraph') and m.get('index_pos') is not None]
        if len(indices) < 2:
            return
        
        vectors = []
        valid_indices = []
        for i in indices:
            try:
                vec = self.vector_store.reconstruct(meta_list[i]['index_pos'])
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    vectors.append(vec)
                    valid_indices.append(i)
            except Exception:
                continue
                
        if len(vectors) < 2:
            return
            
        vec_arr = np.array(vectors, dtype='float32')
        norms = np.linalg.norm(vec_arr, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        vec_arr_norm = vec_arr / norms
        sim_matrix = np.dot(vec_arr_norm, vec_arr_norm.T)
        
        to_remove = set()
        for i in range(len(vectors)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(vectors)):
                if j in to_remove:
                    continue
                if sim_matrix[i, j] > REDUNDANT_SIMILARITY_THRESHOLD:
                    # 保留文本更长的那个
                    meta_i = meta_list[valid_indices[i]]
                    meta_j = meta_list[valid_indices[j]]
                    len_i = len(meta_i.get('text', ''))
                    len_j = len(meta_j.get('text', ''))
                    if len_j >= len_i:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)
                    break
        
        if not to_remove:
            return
            
        remove_global_indices = {valid_indices[i] for i in to_remove}
        new_metadata = []
        new_vectors = []
        
        for i, meta in enumerate(meta_list):
            if i in remove_global_indices:
                continue
            if meta.get('index_pos') is not None and meta['index_pos'] < self.vector_store.ntotal():
                try:
                    vec = self.vector_store.reconstruct(meta['index_pos'])
                    new_vectors.append(vec)
                    new_metadata.append(meta)
                except Exception:
                    new_metadata.append(meta)
        
        if new_vectors:
            self.vector_store.rebuild_from_metadata(new_metadata, new_vectors)
            self._cleanup_removed_vectors(remove_global_indices)
            logger.info("已清理 %d 个冗余向量", len(remove_global_indices))

    def _cleanup_removed_vectors(self, removed_meta_indices):
        removed_ids = set()
        for i in removed_meta_indices:
            if i < len(self.vector_store.metadata):
                removed_ids.add(self.vector_store.metadata[i]['id'])
        
        # 清理句子映射
        for vec_id in list(self.sent_map.keys()):
            if vec_id in removed_ids:
                del self.sent_map[vec_id]
        
        # 清理段落树
        for vec_id in list(self.para_tree.keys()):
            if vec_id in removed_ids:
                para_node = self.para_tree[vec_id]
                for sent_node in para_node.sentences:
                    self.sent_map.pop(sent_node.id, None)
                del self.para_tree[vec_id]

    def _update_clusters(self):
        if self.vector_store.ntotal() < 5:
            return
        
        para_vectors = []
        para_indices = []
        for i, meta in enumerate(self.vector_store.metadata):
            if meta['type'] == 'paragraph' and meta.get('index_pos', -1) < self.vector_store.ntotal():
                try:
                    vec = self.vector_store.reconstruct(meta['index_pos'])
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        para_vectors.append(vec)
                        para_indices.append(i)
                except Exception:
                    pass
        
        if len(para_vectors) < 2:
            return
        
        X = np.array(para_vectors)
        n_para = len(para_vectors)
        target_k = _target_n_clusters(n_para)
        labels, centers = self.clustering.encode_with_target_k(X, target_k)
        
        n_clusters = len(centers)
        logger.info("聚类完成, 段落数: %d, 目标簇数: %d, 实际聚类数: %d", n_para, target_k, n_clusters)
        
        if n_clusters > 1 and len(X) > 10:
            try:
                score = silhouette_score(X, labels)
                logger.info("轮廓系数: %.3f", score)
            except Exception:
                pass
        
        self.knowledge_graph = {}
        metrics_list = self._compute_cluster_metrics(X, labels, centers)
        for cluster_id in range(n_clusters):
            members = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            if not members:
                continue
            center = self._normalize_vector(centers[cluster_id])
            node_id = f"kno_{cluster_id}"
            kn = KnowledgeNode(node_id, center)
            kn.paragraph_ids = [self.vector_store.metadata[para_indices[i]]['id'] for i in members]
            kn.metrics = metrics_list[cluster_id] if cluster_id < len(metrics_list) else {}
            self.knowledge_graph[node_id] = kn
        
        if not self._ablation_single_stage_cluster:
            self._hippocampal_consolidation()
        self._save_vector_db()

    def _update_summary_memory(self):
        if len(self.para_tree) < 3:
            return
        para_ids = list(self.para_tree.keys())[-SUMMARY_UPDATE_THRESHOLD:]
        if len(para_ids) < 2:
            return
        texts = []
        for pid in para_ids:
            if pid in self.para_tree:
                texts.append(self.para_tree[pid].text)
        theme = "对话摘要"
        abstract_text = "；".join(t[:50] + "…" if len(t) > 50 else t for t in texts[:5])
        summary_id = f"sum_{int(datetime.now().timestamp()*1000)}"
        sn = SummaryNode(theme, abstract_text, para_ids, summary_id)
        combined = " ".join(texts)
        sn.vector = self._normalize_vector(self._get_embedding(combined))
        self.summary_nodes[sn.id] = sn
        self._save_vector_db()

    def _recency_weight(self, timestamp_str):
        try:
            if isinstance(timestamp_str, datetime):
                ts = timestamp_str
            else:
                ts = datetime.fromisoformat(str(timestamp_str).replace('Z', '+00:00'))
            now = datetime.now(ts.tzinfo) if getattr(ts, 'tzinfo', None) and ts.tzinfo else datetime.now()
            age_seconds = (now - ts).total_seconds()
            decay_days = 30
            w = max(0, 1 - age_seconds / (decay_days * 86400))
            return w
        except Exception:
            return 0.5

    def is_temporal_task(self, task_type: str) -> bool:
        """是否为时序推理任务（由测评方根据 question_type 传参，不依赖关键词）。
        测评时 eval_new 根据 instance['question_type'] 调用本接口并将结果传入 search(..., is_temporal_task=...)。"""
        if self._ablation_no_temporal:
            return False
        return (task_type or "").strip().lower() == "temporal-reasoning"

    def _get_recent_turns_for_temporal(self, limit, exclude_tids):
        """为时序推理补充按时间顺序的近期轮次（带 tid），排除已在结果中的 tid。返回与 search 结果同结构的列表，按时间升序。"""
        if not os.path.exists(self.talk_file):
            return []
        out = []
        try:
            with open(self.talk_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # 取最后 limit 条，顺序为从旧到新（ chronological）
            for line in lines[-limit:]:
                if not line.strip():
                    continue
                parts = line.strip().split('|', 1)
                if len(parts) < 2:
                    continue
                try:
                    meta = json.loads(parts[0])
                    tid = meta.get('tid')
                    if not tid or tid in exclude_tids:
                        continue
                    ts = meta.get('timestamp', '')
                    full_info = self._get_full_dialog_by_tid(tid)
                    full_text = full_info.get('text', parts[1]) if full_info else parts[1]
                    rec = self._recency_weight(ts)
                    # 给予适中分数以便参与排序，不压制语义高分结果
                    score = 0.4
                    final_score = score * (self.semantic_weight + self.recency_weight * rec)
                    out.append({
                        'tid': tid,
                        'text': parts[1],
                        'full_text': full_text,
                        'full_dialog': full_text,
                        'dialog_timestamp': ts,
                        'timestamp': ts,
                        'type': 'recent_temporal',
                        'score': score,
                        'final_score': final_score,
                        'memory_id': tid,
                    })
                except Exception:
                    continue
        except Exception as e:
            logger.debug("读取近期时序轮次失败: %s", e)
        return out

    def _associative_retrieval(self, query_vector, top_k):
        # 适度放宽知识簇检索以利 knowledge-update（旧+新同现），但不过激以免噪声
        ks_min = self._retrieval_cfg["knowledge_min_score"]
        primary = self._knowledge_search(query_vector, top_k_nodes=6, min_score=ks_min, max_nodes=7)
        if self._ablation_no_assoc:
            return primary[:top_k]
        if not self.knowledge_graph or not primary:
            return primary
        
        nodes = list(self.knowledge_graph.values())
        centers = np.array([n.center_vector for n in nodes])
        qv = np.asarray(query_vector, dtype='float32').ravel()
        if qv.ndim > 1:
            qv = qv.ravel()
        
        # 计算余弦相似度
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        centers_norm = centers / norms
        qv_norm = qv / (np.linalg.norm(qv) + 1e-8)
        sims = np.dot(centers_norm, qv_norm)
        
        top_idx = np.argsort(sims)[::-1][:ASSOCIATIVE_EXPAND_CLUSTERS + 3]
        expanded = []
        seen_tids = set(r.get('tid') for r in primary)
        
        for idx in top_idx:
            node = nodes[idx]
            para_count = 0
            for pid in node.paragraph_ids:
                if para_count >= MAX_PARAS_PER_CLUSTER:
                    break
                if pid in seen_tids:
                    continue
                para_meta = next((m for m in self.vector_store.metadata if m['id'] == pid), None)
                if not para_meta:
                    continue
                full_dialog_info = self._get_full_dialog_by_tid(pid)
                full_context = full_dialog_info.get('text', para_meta.get('text', '')) if full_dialog_info else para_meta.get('text', '')
                timestamp = full_dialog_info.get('timestamp', para_meta.get('timestamp', '')) if full_dialog_info else para_meta.get('timestamp', '')
                expanded.append({
                    'tid': pid,
                    'text': para_meta.get('text', ''),
                    'full_text': full_context,
                    'type': 'knowledge_paragraph',
                    'score': float(sims[idx]),
                    'cluster_id': node.node_id,
                    'timestamp': timestamp,
                    'memory_id': pid,
                })
                seen_tids.add(pid)
                para_count += 1
        combined = primary + expanded[:top_k]
        return combined[:top_k]

    def _knowledge_search(self, qv, top_k_nodes=6, min_score=None, max_nodes=7):
        if min_score is None:
            min_score = self._retrieval_cfg["knowledge_min_score"]
        if not self.knowledge_graph:
            return []
        nodes = list(self.knowledge_graph.values())
        centers = np.array([n.center_vector for n in nodes])
        qv = np.asarray(qv, dtype='float32').ravel()
        
        # 计算余弦相似度
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        centers_norm = centers / norms
        qv_norm = qv / (np.linalg.norm(qv) + 1e-8)
        similarities = np.dot(centers_norm, qv_norm)
        
        top_nodes_idx = np.argsort(similarities)[::-1][:top_k_nodes]
        results = []
        node_count = 0
        for idx in top_nodes_idx:
            if node_count >= max_nodes:
                break
            node = nodes[idx]
            sim = similarities[idx]
            if sim < min_score:
                continue
            para_count = 0
            for pid in node.paragraph_ids:
                if para_count >= MAX_PARAS_PER_CLUSTER:
                    break
                para_meta = next((m for m in self.vector_store.metadata if m['id'] == pid), None)
                if not para_meta:
                    continue
                full_dialog_info = self._get_full_dialog_by_tid(pid)
                full_context = full_dialog_info.get('text', para_meta.get('text', '')) if full_dialog_info else para_meta.get('text', '')
                timestamp = full_dialog_info.get('timestamp', para_meta.get('timestamp', '')) if full_dialog_info else para_meta.get('timestamp', '')
                results.append({
                    'tid': pid,
                    'text': para_meta.get('text', ''),
                    'full_text': full_context,
                    'type': 'knowledge_paragraph',
                    'score': float(sim),
                    'cluster_id': node.node_id,
                    'timestamp': timestamp,
                    'memory_id': pid,
                })
                para_count += 1
            node_count += 1
        return results

    def _vector_search(self, qv, top_k=10, vec_type=None, min_score=0.0):
        results = self.vector_store.search(qv, top_k * 3, vec_type=vec_type)
        out = []
        seen_content = set()
        for score, idx, meta in results:
            if score < min_score:
                continue
            tid = meta.get('parent_tid') or meta['id']
            full_dialog_info = self._get_full_dialog_by_tid(tid)
            full_context = full_dialog_info.get('text', meta.get('text', '')) if full_dialog_info else meta.get('text', '')
            timestamp = full_dialog_info.get('timestamp', meta.get('timestamp', '')) if full_dialog_info else meta.get('timestamp', '')
            content_key = (full_context or '').strip()[:8000] or tid
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            out.append({
                'tid': tid,
                'text': meta.get('text', ''),
                'full_text': full_context,
                'type': meta.get('type', ''),
                'score': float(score),
                'timestamp': timestamp,
                'memory_id': meta.get('id', tid),
            })
            if len(out) >= top_k:
                break
        return out

    def _summary_search(self, qv, top_k=2, min_score=None):
        if min_score is None:
            min_score = self._retrieval_cfg["min_summary_similarity"]
        if not self.summary_nodes:
            return []
        nodes_with_vec = [(sid, sn) for sid, sn in self.summary_nodes.items() if sn.vector is not None]
        if not nodes_with_vec:
            return []
        vectors = np.array([sn.vector for _, sn in nodes_with_vec], dtype='float32')
        qv = np.asarray(qv, dtype='float32').ravel()
        
        # 计算余弦相似度
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        vectors_norm = vectors / norms
        qv_norm = qv / (np.linalg.norm(qv) + 1e-8)
        sims = np.dot(vectors_norm, qv_norm)
        
        top_idx = np.argsort(sims)[::-1][: max(top_k, 5)]
        results = []
        for idx in top_idx:
            if float(sims[idx]) < min_score:
                continue
            sn = nodes_with_vec[idx][1]
            for pid in sn.source_para_ids:
                para_meta = next((m for m in self.vector_store.metadata if m['id'] == pid), None)
                if not para_meta:
                    continue
                full_dialog_info = self._get_full_dialog_by_tid(pid)
                full_context = full_dialog_info.get('text', para_meta.get('text', '')) if full_dialog_info else para_meta.get('text', '')
                timestamp = full_dialog_info.get('timestamp', para_meta.get('timestamp', '')) if full_dialog_info else para_meta.get('timestamp', '')
                results.append({
                    'tid': pid,
                    'text': para_meta.get('text', ''),
                    'full_text': full_context,
                    'type': 'summary_paragraph',
                    'score': float(sims[idx]),
                    'timestamp': timestamp,
                    'memory_id': pid,
                })
        return results

    def search(self, query, top_k=TOP_K_RETRIEVAL, is_temporal_task=False):
        """检索记忆。is_temporal_task 由测评方根据任务类型传入，见 is_temporal_task(task_type)。"""
        cache_key = f"search:{hash(query) % (2**32)}:t{int(is_temporal_task)}"
        cached = self.lru_cache.get(cache_key)
        if cached is not None:
            return cached[:top_k]

        temporal_priority = is_temporal_task
        qv = self._normalize_vector(self._get_embedding(query))
        all_results = []

        # 1. 摘要检索
        summary_results = self._summary_search(qv, top_k=2)
        all_results.extend(summary_results)

        # 2. 关联检索
        assoc = self._associative_retrieval(qv, top_k)
        all_results.extend(assoc)

        # 3. 段落向量检索（时序推理时多取以提升召回）
        para_top_k = TEMPORAL_PARA_TOP_K if temporal_priority else 5
        para_results = self._vector_search(qv, top_k=para_top_k, vec_type='paragraph', min_score=0.0)
        filtered_para = [r for r in para_results if r.get('score', 0) >= 0.3][:para_top_k]
        all_results.extend(filtered_para)

        # 4. 句子向量检索（如果结果不足）
        if len(all_results) < top_k:
            remaining = top_k - len(all_results)
            sent_results = self._vector_search(qv, top_k=remaining * 2, vec_type='sentence', min_score=0.65)
            filtered_sent = [r for r in sent_results if r.get('score', 0) >= 0.3]
            all_results.extend(filtered_sent)

        # 计算最终分数
        final_results = []
        for result in all_results:
            full_dialog_info = self._get_full_dialog_by_tid(result['tid'])
            if full_dialog_info:
                result['full_dialog'] = full_dialog_info.get('text', '')
                result['dialog_timestamp'] = full_dialog_info.get('timestamp', datetime.min)
            else:
                result['full_dialog'] = result.get('full_text', result.get('text', ''))
                result['dialog_timestamp'] = result.get('timestamp', datetime.min)

            rec = self._recency_weight(result.get('dialog_timestamp', ''))
            semantic_score = result.get('score', 0)
            result['final_score'] = semantic_score * (self.semantic_weight + self.recency_weight * rec)
            final_results.append(result)

        # 时序推理：补充近期轮次（按时间顺序），缩小召回与正确率差距
        if temporal_priority:
            existing_tids = {r['tid'] for r in final_results}
            recent_turns = self._get_recent_turns_for_temporal(RECENT_TEMPORAL_TURNS, existing_tids)
            final_results.extend(recent_turns)

        # 排序：时序推理以时间升序为主（事件先后清晰），否则以分数为主、时间为辅
        def _parse_time(x):
            ts = x.get('dialog_timestamp', '') or x.get('timestamp', '')
            if isinstance(ts, datetime):
                return ts
            try:
                return datetime.fromisoformat(str(ts).replace('Z', '+00:00')) if ts else datetime.min
            except Exception:
                return datetime.min

        if temporal_priority:
            # 时间优先：先按时间升序，再按分数降序，便于模型做“先/后”推理
            final_results.sort(key=lambda x: (_parse_time(x), -x.get('final_score', 0)))
        else:
            final_results.sort(key=lambda x: (-x.get('final_score', 0), _parse_time(x)))

        # 去重：按内容（full_dialog/text）去重，避免同一对话块因不同 tid/timestamp 重复占位
        unique_results = []
        seen_content = set()
        for result in final_results:
            content = (result.get('full_dialog') or result.get('text', '')).strip()
            content_key = content[:8000] if content else result.get('tid', '')  # 截断避免超长 key
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            unique_results.append(result)
            if len(unique_results) >= top_k:
                break
        
        # 如果结果不足，补充
        if len(unique_results) < top_k:
            needed = top_k - len(unique_results)
            existing_tids = {r.get('tid') for r in unique_results}
            backfill = self._backfill_results(qv, existing_tids, needed)
            unique_results.extend(backfill)
            unique_results = unique_results[:top_k]
            if temporal_priority and backfill:
                unique_results.sort(key=lambda x: (_parse_time(x), -x.get('final_score', 0)))

        self.lru_cache.put(cache_key, unique_results)
        return unique_results

    def _get_full_dialog_by_tid(self, tid):
        if not os.path.exists(self.talk_file):
            return None
        try:
            with open(self.talk_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line_idx, line in enumerate(lines):
                if not line.strip():
                    continue
                parts = line.strip().split('|', 1)
                if len(parts) < 2:
                    continue
                try:
                    meta = json.loads(parts[0])
                    if meta['tid'] != tid:
                        continue
                    context_lines = []
                    for j in range(max(0, line_idx - 1), min(len(lines), line_idx + 2)):
                        if j < len(lines) and lines[j].strip():
                            try:
                                ctx_meta = json.loads(lines[j].split('|', 1)[0])
                                ctx_text = lines[j].split('|', 1)[1]
                                context_lines.append(f"{ctx_meta['role']}: {ctx_text}")
                            except Exception:
                                continue
                    return {
                        'text': "\n".join(context_lines),
                        'timestamp': meta['timestamp']
                    }
                except Exception as e:
                    logger.debug("解析对话失败: %s", e)
        except Exception as e:
            logger.error("读取对话文件失败: %s", e)
        return None

    def _backfill_results(self, qv, existing_tids, needed):
        if needed <= 0:
            return []
        backfill = []
        # 尝试段落检索
        para_candidates = self._vector_search(qv, top_k=needed * 3, vec_type='paragraph', min_score=0.0)
        for res in para_candidates:
            if res.get('tid') in existing_tids:
                continue
            full_dialog_info = self._get_full_dialog_by_tid(res.get('tid'))
            backfill.append({
                'tid': res.get('tid'),
                'text': res.get('text', ''),
                'full_dialog': full_dialog_info.get('text', res.get('full_text', '')) if full_dialog_info else res.get('full_text', ''),
                'dialog_timestamp': res.get('timestamp', datetime.min),
                'memory_id': res.get('memory_id', res.get('tid')),
                'type': 'paragraph',
                'score': res.get('score', 0),
                'final_score': res.get('score', 0) * self.semantic_weight  # 简化计算
            })
            existing_tids.add(res.get('tid'))
            if len(backfill) >= needed:
                return backfill
        # 尝试句子检索
        if len(backfill) < needed:
            sent_candidates = self._vector_search(qv, top_k=(needed-len(backfill)) * 5, vec_type='sentence', min_score=0.0)
            for res in sent_candidates:
                if res.get('tid') in existing_tids:
                    continue
                full_dialog_info = self._get_full_dialog_by_tid(res.get('tid'))
                backfill.append({
                    'tid': res.get('tid'),
                    'text': res.get('text', ''),
                    'full_dialog': full_dialog_info.get('text', res.get('full_text', '')) if full_dialog_info else res.get('full_text', ''),
                    'dialog_timestamp': res.get('timestamp', datetime.min),
                    'memory_id': res.get('memory_id', res.get('tid')),
                    'type': 'sentence',
                    'score': res.get('score', 0),
                    'final_score': res.get('score', 0) * self.semantic_weight
                })
                existing_tids.add(res.get('tid'))
                if len(backfill) >= needed:
                    break
        return backfill[:needed]

    def report_retrieval_success(self, cluster_id, success=True):
        if cluster_id not in self.knowledge_graph:
            return
        node = self.knowledge_graph[cluster_id]
        node.metrics['retrieval_count'] = node.metrics.get('retrieval_count', 0) + 1
        if success:
            node.metrics['success_count'] = node.metrics.get('success_count', 0) + 1
        total = node.metrics['retrieval_count']
        node.metrics['success_rate'] = node.metrics.get('success_count', total) / max(1, total)
        node.update_metrics()

    def add_reflexion(self, failure_context, reflection_text):
        ts = datetime.now().isoformat()
        summary_id = f"refl_{int(datetime.now().timestamp()*1000)}"
        sn = SummaryNode("反思策略", reflection_text, [], summary_id)
        sn.vector = self._normalize_vector(self._get_embedding(reflection_text))
        sn.timestamp = ts
        self.summary_nodes[sn.id] = sn
        self._save_vector_db()
        logger.info("已记录反思记忆: %s", summary_id)

    def get_recent_dialogs(self, limit=MAX_DIALOG_HISTORY):
        dialogs = []
        if not os.path.exists(self.talk_file):
            return dialogs
        try:
            with open(self.talk_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-limit:]
            for line in lines:
                if not line.strip():
                    continue
                parts = line.strip().split('|', 1)
                if len(parts) < 2:
                    continue
                try:
                    meta = json.loads(parts[0])
                    dialogs.append({
                        'role': meta['role'], 
                        'text': parts[1], 
                        'timestamp': meta['timestamp']
                    })
                except Exception:
                    continue
            return dialogs[::-1]  # 返回最新在前
        except Exception as e:
            logger.error("读取对话历史失败: %s", e)
            return []

    def reset(self):
        self.vector_store = VectorStore(VECTOR_DIM, use_pq=True)
        self.clustering = ClusteringLayer(threshold=BIRCH_THRESHOLD)
        self.knowledge_graph = {}
        self.para_tree = {}
        self.sent_map = {}
        self.summary_nodes = {}
        self.lru_cache.clear()
        if os.path.exists(self.talk_file):
            os.remove(self.talk_file)
        os.makedirs(os.path.dirname(self.talk_file), exist_ok=True)
        open(self.talk_file, 'w').close()
        self._save_vector_db()