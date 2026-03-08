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

    def _consolidate_high_similarity(self):
            """
            【新增功能】
            定期整合向量库中相似度过高的句子和段落向量，减少冗余。
            将相似度极高的向量合并，保留质量最高的一个。
            """
            logger.info("开始整合高相似度句子和段落向量...")
            
            # 收集所有需要检查的向量
            vectors_to_check = []
            meta_indices = []
            
            for i, meta in enumerate(self.vector_metadata):
                if meta['type'] in ['sentence', 'paragraph']:
                    if meta['index_pos'] < self.vector_index.ntotal:
                        try:
                            vec = self.vector_index.reconstruct(meta['index_pos'])
                            if np.linalg.norm(vec) > 1e-6:  # 过滤零向量
                                vectors_to_check.append(vec)
                                meta_indices.append(i)
                        except Exception as e:
                            logger.warning(f"无法获取向量 {meta['index_pos']}: {e}")
                            continue
            
            if len(vectors_to_check) < 2:
                logger.info("向量数量不足，无需整合。")
                return
            
            # 计算向量间的相似度矩阵
            vectors_array = np.array(vectors_to_check)
            similarity_matrix = np.dot(vectors_array, vectors_array.T)
            
            # 计算每个向量的范数用于归一化
            norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
            norms_matrix = norms * norms.T
            norms_matrix = np.where(norms_matrix > 0, norms_matrix, 1)
            similarity_matrix = similarity_matrix / norms_matrix
            
            # 查找高相似度向量对
            high_sim_pairs = []
            seen_indices = set()
            
            for i in range(len(vectors_array)):
                if i in seen_indices:
                    continue
                    
                for j in range(i+1, len(vectors_array)):
                    if j in seen_indices:
                        continue
                        
                    if similarity_matrix[i, j] > 0.9:  # 相似度阈值设为0.9
                        # 获取对应的元数据
                        meta_i = self.vector_metadata[meta_indices[i]]
                        meta_j = self.vector_metadata[meta_indices[j]]
                        
                        # 计算质量分数（基于文本长度和向量范数）
                        quality_i = len(meta_i['text']) + np.linalg.norm(vectors_array[i])
                        quality_j = len(meta_j['text']) + np.linalg.norm(vectors_array[j])
                        
                        # 保留质量更高的
                        if quality_i >= quality_j:
                            high_sim_pairs.append((j, i))  # 合并j到i
                            seen_indices.add(j)
                        else:
                            high_sim_pairs.append((i, j))  # 合并i到j
                            seen_indices.add(i)
                        break
            
            if not high_sim_pairs:
                logger.info("未发现需要整合的高相似度向量。")
                return
            
            logger.info(f"发现 {len(high_sim_pairs)} 个高相似度向量对，正在整合...")
            
            # 确定需要保留和移除的索引
            to_remove_indices = set()
            for remove_idx, keep_idx in high_sim_pairs:
                meta_remove = self.vector_metadata[meta_indices[remove_idx]]
                meta_keep = self.vector_metadata[meta_indices[keep_idx]]
                
                logger.debug(f"合并: '{meta_remove['text'][:20]}...' -> '{meta_keep['text'][:20]}...'")
                to_remove_indices.add(meta_indices[remove_idx])
            
            # 创建新的元数据和向量列表
            new_metadata = []
            new_vectors = []
            
            for i, meta in enumerate(self.vector_metadata):
                if i in to_remove_indices:
                    continue  # 跳过要移除的
                    
                if meta['index_pos'] < self.vector_index.ntotal:
                    try:
                        vec = self.vector_index.reconstruct(meta['index_pos'])
                        new_vectors.append(vec)
                        new_metadata.append(meta)
                    except Exception as e:
                        logger.warning(f"重建索引时无法获取向量 {meta['index_pos']}: {e}")
                        new_metadata.append(meta)  # 保留元数据
            
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
                
                logger.info(f"高相似度向量整合完成，移除了 {len(to_remove_indices)} 个冗余向量，索引已重建。")
                
                # 更新内存中的数据结构
                self._cleanup_removed_vectors(to_remove_indices)
            else:
                logger.warning("没有有效向量可用于重建索引。")
        
    def _cleanup_removed_vectors(self, removed_meta_indices):
        """清理被移除的向量对应的内存结构"""
        removed_ids = []
        for i in removed_meta_indices:
            if i < len(self.vector_metadata):
                meta = self.vector_metadata[i]
                removed_ids.append(meta['id'])
        
        # 从sent_map和para_tree中清理
        for vec_id in removed_ids:
            if vec_id in self.sent_map:
                del self.sent_map[vec_id]
            
            # 如果是段落，清理para_tree
            if vec_id in self.para_tree:
                # 先清理段落中的句子
                para_node = self.para_tree[vec_id]
                for sent_node in para_node.sentences:
                    if sent_node.id in self.sent_map:
                        del self.sent_map[sent_node.id]
                del self.para_tree[vec_id]
        
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
            center_normalized = self._normalize_vector(center)  # 新增归一化
            node_id = f"kno_{cluster_id}"
            kn = KnowledgeNode(node_id, center_normalized)  # 使用归一化向量
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

        # 【修改】在聚类更新后，执行一次高相似度向量整合
        self._consolidate_high_similarity()
        
        self._save_vector_db()


    def search(self, query, top_k=TOP_K_RETRIEVAL):
        """
        优化搜索逻辑：知识级搜索（无相似度筛选）→ 段落级搜索 → 句子级搜索
        返回完整的对话对，包含user和assistant部分
        """
        qv = self._get_embedding(query)
        qv = self._normalize_vector(qv)
        
        all_results = []
        
        # 1. 知识级搜索 - 无相似度筛选
        kg_results = self._knowledge_search(qv, top_k=5)
        filtered_kg_results = [r for r in kg_results if r.get('score', 0) >= 0.5][:3]  # 仅保留高相关的知识节点
        all_results.extend(filtered_kg_results)
        print(f"知识级搜索结果: {len(kg_results)} -> 筛选后: {len(filtered_kg_results)}")
        
        # 2. 段落级搜索 - 应用相似度筛选
        para_results = self._vector_search(qv, top_k=5, vec_type='paragraph', min_score=0.0)
        filtered_para_results = [r for r in para_results if r.get('score', 0) >= 0.3][:5]
        all_results.extend(filtered_para_results)
        print(f"段落级搜索结果: {len(para_results)} -> 筛选后: {len(filtered_para_results)}")
        
        # 3. 句子级搜索 - 仅当结果不足时启用，并应用相似度筛选
        if len(all_results) < top_k:
            remaining = top_k - len(all_results)
            sent_results = self._vector_search(qv, top_k=remaining*2, vec_type='sentence', min_score=0.65)
            filtered_sent_results = [r for r in sent_results if r.get('score', 0) >= 0.3]
            all_results.extend(filtered_sent_results)
        print(f"句子级搜索结果: {len(sent_results)} -> 筛选后: {len(filtered_sent_results)}")
        
        # 获取完整对话上下文
        final_results = []
        for result in all_results:
            # 获取完整对话
            full_dialog = self._get_full_dialog_by_tid(result['tid'])
            
            if full_dialog:
                # 如果返回的是字符串格式的完整对话，直接存储
                result['full_dialog'] = full_dialog
            else:
                # 如果没有找到完整对话，使用当前文本
                result['full_dialog'] = result.get('text', '')
            
            final_results.append(result)
        
        # 去重和排序
        unique_results = self._deduplicate_and_sort(final_results, top_k)
        return unique_results

    def _knowledge_search(self, qv, top_k=5):
        if not self.knowledge_graph:
            return []
        
        nodes = list(self.knowledge_graph.values())
        print(nodes)
        centers = np.array([n.center_vector for n in nodes])
        print(centers.shape, qv.shape)
        
        # 计算余弦相似度而不是欧式距离
        similarities = np.dot(centers, qv)  # 已归一化，直接点积就是余弦相似度
        print(similarities)
        
        top_nodes_idx = np.argsort(similarities)[::-1][:top_k]  # 降序排序
        
        results = []
        for idx in top_nodes_idx:
            node = nodes[idx]
            similarity = similarities[idx]
            
            # 只保留有足够相似度的结果
            if similarity < 0.3:  # 设置阈值
                continue
                
            for pid in node.paragraph_ids:
                meta = next((m for m in self.vector_metadata if m['id'] == pid), None)
                if meta:
                    full_context = self._get_full_dialog_by_tid(pid)
                    results.append({
                        'tid': pid,
                        'text': meta['text'],
                        'full_text': full_context or meta['text'],
                        'type': 'knowledge_item',
                        'score': float(similarity),  # 使用余弦相似度分数
                        'cluster_id': node.node_id
                    })
        
        return results

    def _vector_search(self, qv, top_k=10, vec_type=None, min_score=0.0):
        """增强版向量搜索，支持类型过滤和最低分数，并返回完整对话"""
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
            
            # 获取tid（句子级用parent_tid，段落级用自身id）
            tid = meta.get('parent_tid') or meta['id']
            
            # 获取完整对话上下文
            full_context = self._get_full_dialog_by_tid(tid)
            if not full_context:
                full_context = meta['text']
                
            # 去重处理 (基于对话对hash)
            context_hash = hash(full_context)
            if context_hash not in seen:
                seen.add(context_hash)
                results.append({
                    'tid': tid,
                    'text': meta['text'],
                    'full_text': full_context,  # 包含完整对话
                    'type': meta['type'],
                    'score': float(scores[0][i])
                })
                
            if len(results) >= top_k:
                break
                
        return results

    def _get_full_dialog_by_tid(self, tid):
        """获取完整的对话对（包含查询结果的前后对话）"""
        if not os.path.exists(self.talk_file):
            return None
            
        # 获取当前对话记录
        with open(self.talk_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 查找匹配的对话
        for i, line in enumerate(lines):
            if line.strip():
                parts = line.strip().split('|', 1)
                if len(parts) == 2:
                    try:
                        meta = json.loads(parts[0])
                        if meta['tid'] == tid:
                            # 获取上下文对话，构建完整的对话对
                            dialog_pairs = []
                            
                            # 向前查找：最多2条对话
                            for j in range(max(0, i-1), i+1):
                                if j < len(lines) and lines[j].strip():
                                    pair_parts = lines[j].strip().split('|', 1)
                                    if len(pair_parts) == 2:
                                        try:
                                            pair_meta = json.loads(pair_parts[0])
                                            dialog_pairs.append({
                                                'role': pair_meta['role'],
                                                'text': pair_parts[1],
                                                'timestamp': pair_meta['timestamp']
                                            })
                                        except:
                                            continue
                            
                            # 向后查找：最多1条对话
                            for j in range(i+1, min(i+2, len(lines))):
                                if j < len(lines) and lines[j].strip():
                                    pair_parts = lines[j].strip().split('|', 1)
                                    if len(pair_parts) == 2:
                                        try:
                                            pair_meta = json.loads(pair_parts[0])
                                            dialog_pairs.append({
                                                'role': pair_meta['role'],
                                                'text': pair_parts[1],
                                                'timestamp': pair_meta['timestamp']
                                            })
                                        except:
                                            continue
                            
                            # 确保至少有当前对话
                            if not dialog_pairs:
                                dialog_pairs.append({
                                    'role': meta['role'],
                                    'text': parts[1],
                                    'timestamp': meta['timestamp']
                                })
                            
                            # 按时间排序
                            dialog_pairs.sort(key=lambda x: x['timestamp'])
                            
                            # 转换为对话格式
                            return self._format_dialog_context(dialog_pairs)
                    except Exception as e:
                        logger.error(f"解析对话失败: {e}")
                        continue
        return None

    def _format_dialog_context(self, dialog_pairs):
        """格式化对话上下文为可读字符串"""
        if not dialog_pairs:
            return ""
        
        formatted = []
        for dialog in dialog_pairs:
            role_name = "用户" if dialog['role'] == 'user' else "助手"
            formatted.append(f"{role_name}: {dialog['text']}")
        
        return "\n".join(formatted)

    def _deduplicate_and_sort(self, results, top_k):
        """去重和排序结果"""
        seen = set()
        unique = []
        
        for result in results:
            # 基于对话内容去重
            if 'full_text' in result:
                text_hash = hash(result['full_text'])
                if text_hash not in seen:
                    seen.add(text_hash)
                    unique.append(result)
        
        # 按分数降序排序
        unique.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return unique[:top_k]

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
