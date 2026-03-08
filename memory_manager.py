import json
import numpy as np
from datetime import datetime
import faiss
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import requests
import os
import logging
import re
from config import *

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 修改默认召回数量为10
TOP_K_RETRIEVAL = 10

# ==================== 日志配置 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== 数据结构定义 ====================
class SentenceNode:
    """句子级节点"""
    def __init__(self, sent_id, text, vector, parent_para_id):
        self.id = sent_id
        self.text = text
        self.vector = vector
        self.parent_para_id = parent_para_id


class ParagraphNode:
    """段落级节点（树形结构根）"""
    def __init__(self, para_id, text):
        self.id = para_id
        self.text = text
        self.para_vector = None  # 整段统一表征向量
        self.sentences = []  # SentenceNode 列表

    def add_sentence(self, sent_node):
        self.sentences.append(sent_node)


class KnowledgeNode:
    """知识级节点（聚类中心 + 关联的段落节点）"""
    def __init__(self, node_id, center_vector):
        self.node_id = node_id
        self.center_vector = center_vector
        self.paragraph_ids = []  # 属于该知识节点的段落ID
        self.related_node_ids = []  # 关联的其他知识节点ID


# ==================== 主管理类 ====================
class VectorMemoryManager:
    def __init__(self):
        self.talk_file = TALK_FILE
        self.vector_index = None
        self.vector_metadata = []  # 所有向量元数据（含树形关系）
        self.cluster_model = None
        self.knowledge_graph = {}  # {node_id: KnowledgeNode}
        self.para_tree = {}  # {para_id: ParagraphNode}
        self.sent_map = {}  # {sent_id: SentenceNode}
        self.cache = {}
        self.dimension_verified = False
        self._initialize_vector_db()

    def _initialize_vector_db(self):
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

        index_path = f"{VECTOR_DB_DIR}/vector.index"
        metadata_path = f"{VECTOR_DB_DIR}/metadata.json"
        knowledge_path = f"{KNOWLEDGE_DIR}/knowledge_graph.json"

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.vector_index = faiss.read_index(index_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.vector_metadata = json.load(f)
            if self.vector_index.d != VECTOR_DIM:
                logger.warning("维度不匹配，重建索引")
                self.vector_index = faiss.IndexFlatIP(VECTOR_DIM)
                self.vector_metadata = []
        else:
            self.vector_index = faiss.IndexFlatIP(VECTOR_DIM)
            self.vector_metadata = []

        # 加载知识图
        if os.path.exists(knowledge_path):
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
                for node_id, data in kg_data.items():
                    kn = KnowledgeNode(node_id, np.array(data['center_vector'], dtype='float32'))
                    kn.paragraph_ids = data['paragraph_ids']
                    kn.related_node_ids = data['related_node_ids']
                    self.knowledge_graph[node_id] = kn

        self.cluster_model = MiniBatchKMeans(n_clusters=10, random_state=42)
        self._update_clusters()

    def _get_embedding(self, text):
        if not text or len(text.strip()) < 3:
            return np.zeros(VECTOR_DIM, dtype='float32')
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
                timeout=10
            )
            if resp.status_code == 200:
                emb = resp.json().get('embedding', [])
                if len(emb) == VECTOR_DIM:
                    return np.array(emb, dtype='float32')
        except Exception as e:
            logger.error(f"获取嵌入失败: {e}")
        return np.random.normal(0, 0.1, VECTOR_DIM).astype('float32')

    def _generate_tid(self):
        return f"tid_{int(datetime.now().timestamp()*1000)}"

    def _split_sentences(self, text):
        # 多符号分句
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

    def add_dialog(self, role, text):
        tid_para = self._generate_tid()
        # 1. 写 talk.txt
        try:
            with open(self.talk_file, 'a', encoding='utf-8') as f:
                meta = {
                    'tid': tid_para,
                    'timestamp': datetime.now().isoformat(),
                    'role': role
                }
                f.write(f"{json.dumps(meta)}|{text}\n")
        except Exception as e:
            logger.error(f"保存对话失败: {e}")

        # 2. 段落级
        para_node = ParagraphNode(tid_para, text)
        para_vector = self._get_embedding(text)
        para_node.para_vector = para_vector
        self.para_tree[tid_para] = para_node

        # 3. 句子级
        sentences = self._split_sentences(text)
        for i, sent in enumerate(sentences):
            sent_id = f"{tid_para}_sent{i}"
            sent_vector = self._get_embedding(sent)
            sent_node = SentenceNode(sent_id, sent, sent_vector, tid_para)
            para_node.add_sentence(sent_node)
            self.sent_map[sent_id] = sent_node
            self._add_to_vector_db(sent_id, sent_vector, sent, 'sentence', tid_para)

        # 4. 段落向量入库
        self._add_to_vector_db(tid_para, para_vector, text, 'paragraph', None)

        # 5. 定期更新聚类与知识图
        if len(self.vector_metadata) % CLUSTER_UPDATE_THRESHOLD == 0:
            self._update_clusters()

        return tid_para

    def _add_to_vector_db(self, vec_id, vector, text, vec_type, parent_tid):
        if len(vector) != VECTOR_DIM:
            vector = np.resize(vector, VECTOR_DIM)
        norm = np.linalg.norm(vector)
        if norm > 1e-8:
            vector = vector / norm
        else:
            vector = np.random.normal(0, 0.1, VECTOR_DIM).astype('float32')
            vector = vector / np.linalg.norm(vector)

        arr = np.array([vector], dtype='float32')
        self.vector_index.add(arr)
        new_idx = self.vector_index.ntotal - 1

        meta = {
            'id': vec_id,
            'text': text,
            'type': vec_type,
            'parent_tid': parent_tid,
            'timestamp': datetime.now().isoformat(),
            'index_pos': new_idx
        }
        self.vector_metadata.append(meta)
        self._save_vector_db()

    def _save_vector_db(self):
        faiss.write_index(self.vector_index, f"{VECTOR_DB_DIR}/vector.index")
        with open(f"{VECTOR_DB_DIR}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.vector_metadata, f, ensure_ascii=False, indent=2)
        # 保存知识图
        kg_data = {}
        for nid, node in self.knowledge_graph.items():
            kg_data[nid] = {
                'center_vector': node.center_vector.tolist(),
                'paragraph_ids': node.paragraph_ids,
                'related_node_ids': node.related_node_ids
            }
        with open(f"{KNOWLEDGE_DIR}/knowledge_graph.json", 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=2)

    def _update_clusters(self):
        if len(self.vector_metadata) < 10:
            return
        para_vectors = []
        para_indices = []
        for meta in self.vector_metadata:
            if meta['type'] == 'paragraph':
                try:
                    vec = self.vector_index.reconstruct(meta['index_pos'])
                    if np.linalg.norm(vec) > 1e-6:
                        para_vectors.append(vec)
                        para_indices.append(meta['index_pos'])
                except Exception:
                    pass
        if len(para_vectors) < 2:
            return
        X = np.array(para_vectors)
        n_clusters = min(10, max(2, len(para_vectors) // 5))
        self.cluster_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        labels = self.cluster_model.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            logger.info(f"聚类完成, 轮廓系数: {score:.3f}, 聚类数: {n_clusters}")

        # 构建知识图
        for cluster_id in range(n_clusters):
            members = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            if not members:
                continue
            center = self.cluster_model.cluster_centers_[cluster_id]
            node_id = f"kno_{cluster_id}"
            kn = KnowledgeNode(node_id, center)
            kn.paragraph_ids = [self.vector_metadata[para_indices[i]]['id'] for i in members]
            self.knowledge_graph[node_id] = kn

        # 建立知识节点间关联
        nodes = list(self.knowledge_graph.values())
        for i in range(len(nodes)):
            sims = []
            for j in range(len(nodes)):
                if i != j:
                    sim = np.dot(nodes[i].center_vector, nodes[j].center_vector) / (
                        np.linalg.norm(nodes[i].center_vector) * np.linalg.norm(nodes[j].center_vector) + 1e-8)
                    sims.append((j, sim))
            sims.sort(key=lambda x: x[1], reverse=True)
            nodes[i].related_node_ids = [nodes[j].node_id for j, _ in sims[:3]]
        self._save_vector_db()

    def search(self, query, top_k=TOP_K_RETRIEVAL):
        qv = self._get_embedding(query)
        # 知识层
        kg_res = self._knowledge_search(qv, top_k=5)
        # 数据层
        data_res = self._vector_search(qv, top_k=10)
        # 融合
        final = self._merge_results({'knowledge': kg_res, 'data': data_res}, query, top_k)
        return final

    def _knowledge_search(self, qv, top_k=5):
        if not self.knowledge_graph:
            return []
        qv = qv / (np.linalg.norm(qv) + 1e-8)
        nodes = list(self.knowledge_graph.values())
        centers = np.array([n.center_vector for n in nodes])
        dists = np.linalg.norm(centers - qv, axis=1)
        top_nodes_idx = np.argsort(dists)[:top_k]
        results = []
        for idx in top_nodes_idx:
            node = nodes[idx]
            for pid in node.paragraph_ids:
                meta = next((m for m in self.vector_metadata if m['id'] == pid), None)
                if meta:
                    results.append({
                        'tid': pid,
                        'text': meta['text'],
                        'type': 'knowledge_item',
                        'score': float(-dists[idx]),
                        'cluster_id': node.node_id
                    })
        return results

    def _vector_search(self, qv, top_k=10):
        qv = np.array([qv], dtype='float32')
        norm = np.linalg.norm(qv)
        if norm > 1e-8:
            qv = qv / norm
        scores, indices = self.vector_index.search(qv, top_k * 2)
        results = []
        seen = set()
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.vector_metadata):
                continue
            meta = self.vector_metadata[idx]
            tid = meta.get('parent_tid') or meta['id']
            if tid not in seen and scores[0][i] > 0.3:
                seen.add(tid)
                results.append({
                    'tid': tid,
                    'text': meta['text'],
                    'type': meta['type'],
                    'score': float(scores[0][i])
                })
            if len(results) >= top_k:
                break
        return results

    def _merge_results(self, results, query, max_results=TOP_K_RETRIEVAL):
        all_res = []
        for rtype, items in results.items():
            for it in items:
                full = self._get_full_dialog_by_tid(it['tid'])
                if full:
                    it['full_text'] = full
                    all_res.append(it)
        unique = {}
        for r in sorted(all_res, key=lambda x: x['score'], reverse=True):
            tid = r['tid']
            if tid not in unique or r['score'] > unique[tid]['score']:
                unique[tid] = r
        return list(unique.values())[:max_results]

    def _get_full_dialog_by_tid(self, tid):
        if not os.path.exists(self.talk_file):
            return None
        with open(self.talk_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('|', 1)
                    if len(parts) == 2:
                        try:
                            meta = json.loads(parts[0])
                            if meta['tid'] == tid:
                                return parts[1]
                        except Exception:
                            continue
        return None

    def get_recent_dialogs(self, limit=MAX_DIALOG_HISTORY):
        dialogs = []
        if not os.path.exists(self.talk_file):
            return dialogs
        with open(self.talk_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-limit:]
            for line in lines:
                if line.strip():
                    parts = line.strip().split('|', 1)
                    if len(parts) == 2:
                        try:
                            meta = json.loads(parts[0])
                            dialogs.append({
                                'role': meta['role'],
                                'text': parts[1],
                                'timestamp': meta['timestamp']
                            })
                        except Exception:
                            continue
        return dialogs[::-1]