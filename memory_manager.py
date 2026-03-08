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

# 配置日志 (只保留一处配置)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 修改默认召回数量为10
TOP_K_RETRIEVAL = 10

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
        # 初始化时先尝试更新聚类，以加载已有数据
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

    def _normalize_vector(self, vector):
        """归一化向量"""
        arr = np.array(vector, dtype='float32')
        norm = np.linalg.norm(arr)
        if norm > 1e-8:
            return arr / norm
        else:
            # 返回一个小的随机向量，避免零向量
            return np.random.normal(0, 0.1, VECTOR_DIM).astype('float32')

    def _should_add_sentence_vector(self, sent_vector, para_vector):
        """
        判断句子向量是否应加入向量库。
        过滤掉与段落主题无关的噪音句子。
        """
        if para_vector is None:
            return True
        
        norm_para = np.linalg.norm(para_vector)
        norm_sent = np.linalg.norm(sent_vector)
        
        if norm_para < 1e-8 or norm_sent < 1e-8:
            return False
            
        similarity = np.dot(para_vector, sent_vector) / (norm_para * norm_sent)
        # 相似度必须大于30%，否则视为噪音句子，不加入向量库
        return similarity > 0.3

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
        para_vector = self._normalize_vector(para_vector)
        para_node.para_vector = para_vector
        self.para_tree[tid_para] = para_node

        # 3. 句子级
        sentences = self._split_sentences(text)
        for i, sent in enumerate(sentences):
            sent_id = f"{tid_para}_sent{i}"
            sent_vector = self._get_embedding(sent)
            sent_vector = self._normalize_vector(sent_vector)
            sent_node = SentenceNode(sent_id, sent, sent_vector, tid_para)
            para_node.add_sentence(sent_node)
            self.sent_map[sent_id] = sent_node
            
            # 【核心修改】过滤低质量句子向量，不直接入库
            if self._should_add_sentence_vector(sent_vector, para_vector):
                self._add_to_vector_db(sent_id, sent_vector, sent, 'sentence', tid_para)
            else:
                logger.debug(f"句子向量相似度过低，已跳过入库: '{sent[:15]}...'")

        # 4. 段落向量入库
        self._add_to_vector_db(tid_para, para_vector, text, 'paragraph', None)

        # 5. 定期更新聚类与知识图
        if len(self.vector_metadata) % CLUSTER_UPDATE_THRESHOLD == 0:
            self._update_clusters()

        return tid_para

    def _add_to_vector_db(self, vec_id, vector, text, vec_type, parent_tid):
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

    def _consolidate_low_similarity_sentences(self):
        """
        【新增功能】
        定期整合向量库中相似度极低的句子向量（噪音）。
        将这些句子从FAISS索引中移除，但保留在内存结构中以备他用。
        """
        logger.info("开始整合低相似度句子向量...")
        low_sim_sentences_info = []  # 存储需要移除的句子元数据索引
        
        # 遍历所有句子类型的元数据
        for i, meta in enumerate(self.vector_metadata):
            if meta['type'] == 'sentence':
                sent_id = meta['id']
                if sent_id in self.sent_map:
                    sent_node = self.sent_map[sent_id]
                    para_id = sent_node.parent_para_id
                    
                    if para_id in self.para_tree:
                        para_node = self.para_tree[para_id]
                        para_vector = para_node.para_vector
                        
                        # 计算句子与其所属段落的相似度
                        if para_vector is not None and sent_node.vector is not None:
                            norm_para = np.linalg.norm(para_vector)
                            norm_sent = np.linalg.norm(sent_node.vector)
                            
                            if norm_para > 1e-8 and norm_sent > 1e-8:
                                similarity = np.dot(para_vector, sent_node.vector) / (norm_para * norm_sent)
                                
                                # 如果相似度低于5%，标记为待移除
                                if similarity < 0.05:
                                    logger.debug(f"标记低相似度句子 '{sent_node.text[:15]}...' (相似度: {similarity:.2f}) 待移除")
                                    low_sim_sentences_info.append(i)
        
        if not low_sim_sentences_info:
            logger.info("未发现需要整合的低相似度句子。")
            return
            
        logger.info(f"发现 {len(low_sim_sentences_info)} 个低相似度句子，正在从索引中移除...")

        # 创建新的元数据和向量列表，排除要移除的项
        new_metadata = []
        new_vectors = []
        
        for i, meta in enumerate(self.vector_metadata):
            if i in low_sim_sentences_info:
                continue  # 跳过要移除的
                
            if meta['index_pos'] < self.vector_index.ntotal:
                try:
                    vec = self.vector_index.reconstruct(meta['index_pos'])
                    new_vectors.append(vec)
                    new_metadata.append(meta)
                except Exception as e:
                    logger.warning(f"重建索引时无法获取向量 {meta['index_pos']}: {e}")
                    new_metadata.append(meta)  # 即使没拿到向量也保留元数据，避免断链
        
        # 重建FAISS索引
        if new_vectors:
            dim = len(new_vectors[0])
            new_index = faiss.IndexFlatIP(dim)
            new_index.add(np.array(new_vectors).astype('float32'))
            self.vector_index = new_index
            
            # 更新所有元数据中的索引位置
            for new_i, meta in enumerate(new_metadata):
                meta['index_pos'] = new_i
                
            self.vector_metadata = new_metadata
            self._save_vector_db()
            logger.info("低相似度句子整合完成，索引已重建。")
        else:
            logger.warning("没有有效向量可用于重建索引。")

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

        # 【新增】在聚类更新后，执行一次低相似度句子整合
        self._consolidate_low_similarity_sentences()
        
        self._save_vector_db()

    def search(self, query, top_k=TOP_K_RETRIEVAL):
        """
        【核心修改】
        重构搜索逻辑：优先检索知识级和段落级，最后才检索高质量的句子级。
        """
        qv = self._get_embedding(query)
        qv = self._normalize_vector(qv)

        # 1. 优先知识层搜索 (顶层语义)
        kg_res = self._knowledge_search(qv, top_k=3)

        # 2. 段落级搜索 (中层语义)
        para_res = self._vector_search(qv, top_k=5, vec_type='paragraph', min_score=0.0)

        # 3. 句子级搜索 (底层细节，仅当结果不足时启用，且设高门槛)
        # 如果前两级结果已经足够，就不需要去句子库里找了，因为句子库可能有很多干扰
        combined_res = kg_res + para_res
        if len(combined_res) < top_k:
            remaining = top_k - len(combined_res)
            # 句子级检索设置更高的分数阈值 (0.65)，确保相关性
            sent_res = self._vector_search(qv, top_k=remaining * 2, vec_type='sentence', min_score=0.65)
            combined_res.extend(sent_res)

        # 融合结果
        final = self._merge_results({'knowledge': kg_res, 'data': combined_res}, query, top_k)
        return final

    def _knowledge_search(self, qv, top_k=5):
        if not self.knowledge_graph:
            return []
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
                        'score': float(-dists[idx]),  # 距离转分数
                        'cluster_id': node.node_id
                    })
        return results

    def _vector_search(self, qv, top_k=10, vec_type=None, min_score=0.0):
        """增强版向量搜索，支持类型过滤和最低分数"""
        if self.vector_index.ntotal == 0:
            return []
            
        # 搜索比需要的数量多一些，以便过滤
        search_k = min(top_k * 3, self.vector_index.ntotal)
        scores, indices = self.vector_index.search(np.array([qv], dtype='float32'), search_k)
        
        results = []
        seen = set()
        
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.vector_metadata):
                continue
                
            meta = self.vector_metadata[idx]
            
            # 类型过滤
            if vec_type and meta['type'] != vec_type:
                continue
                
            # 分数过滤 (FAISS IP距离，越高越好)
            if scores[0][i] < min_score:
                continue
                
            # 去重处理 (基于tid)
            tid = meta.get('parent_tid') or meta['id']
            if tid not in seen:
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
        """增强版结果融合，按优先级排序"""
        # 优先级：知识节点 > 段落 > 句子
        priority_order = {'knowledge_item': 1, 'paragraph': 2, 'sentence': 3}
        
        all_res = []
        for rtype, items in results.items():
            for it in items:
                full = self._get_full_dialog_by_tid(it['tid'])
                if full:
                    it['full_text'] = full
                    # 添加优先级信息用于排序
                    it['priority'] = priority_order.get(it['type'], 99)
                    all_res.append(it)
        
        # 按优先级(升序)和分数(降序)排序
        unique = {}
        for r in sorted(all_res, key=lambda x: (-x['priority'], -x['score'])):
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
