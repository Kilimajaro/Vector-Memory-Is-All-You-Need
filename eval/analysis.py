import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入配置
try:
    from config import *
except ImportError:
    # 默认配置
    VECTOR_DB_DIR = "./vector_db"
    KNOWLEDGE_DIR = "./knowledge"
    EMBEDDING_MODEL = "bge-m3"
    OLLAMA_BASE_URL = "http://localhost:11434"
    VECTOR_DIM = 1024
    MAX_DIALOG_HISTORY = 100
    CLUSTER_UPDATE_THRESHOLD = 50
    TALK_FILE = "./talk.txt"


class MemoryAnalyzer:
    """
    记忆分析器 - 对VectorMemoryManager的所有数据进行全景展示和手动修复
    """
    
    def __init__(self, vector_manager=None):
        self.vm = vector_manager
        self.figures = []
        
    def load_from_files(self):
        """从文件加载数据"""
        # 加载向量索引
        index_path = f"{VECTOR_DB_DIR}/vector.index"
        metadata_path = f"{VECTOR_DB_DIR}/metadata.json"
        knowledge_path = f"{KNOWLEDGE_DIR}/knowledge_graph.json"
        talk_path = TALK_FILE
        
        # 加载向量元数据
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.vector_metadata = json.load(f)
        else:
            self.vector_metadata = []
            
        # 加载知识图
        if os.path.exists(knowledge_path):
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                self.knowledge_data = json.load(f)
        else:
            self.knowledge_data = {}
            
        # 加载对话历史
        self.dialogs = []
        if os.path.exists(talk_path):
            with open(talk_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split('|', 1)
                        if len(parts) == 2:
                            try:
                                meta = json.loads(parts[0])
                                self.dialogs.append({
                                    'meta': meta,
                                    'text': parts[1]
                                })
                            except Exception:
                                continue
        
        # 构建内存数据结构
        self._build_memory_structures()
        
    def _build_memory_structures(self):
        """构建内存数据结构"""
        # 按类型分类
        self.sentences = []  # 句子级
        self.paragraphs = []  # 段落级
        self.knowledge_nodes = []  # 知识级
        
        for meta in self.vector_metadata:
            if meta['type'] == 'sentence':
                self.sentences.append(meta)
            elif meta['type'] == 'paragraph':
                self.paragraphs.append(meta)
                
        for node_id, data in self.knowledge_data.items():
            self.knowledge_nodes.append({
                'node_id': node_id,
                'center_vector': data['center_vector'],
                'paragraph_ids': data['paragraph_ids'],
                'related_node_ids': data['related_node_ids']
            })
    
    # ==================== 统计信息 ====================
    
    def print_overview(self):
        """打印概览信息"""
        print("\n" + "=" * 60)
        print("📊 记忆系统全景概览")
        print("=" * 60)
        
        total_vectors = len(self.vector_metadata)
        total_sentences = len(self.sentences)
        total_paragraphs = len(self.paragraphs)
        total_knowledge = len(self.knowledge_nodes)
        total_dialogs = len(self.dialogs)
        
        print(f"\n📁 数据规模:")
        print(f"   • 总向量数: {total_vectors}")
        print(f"   • 句子级向量: {total_sentences}")
        print(f"   • 段落级向量: {total_paragraphs}")
        print(f"   • 知识节点: {total_knowledge}")
        print(f"   • 对话记录: {total_dialogs}")
        
        if self.dialogs:
            roles = Counter(d['meta']['role'] for d in self.dialogs)
            print(f"\n👥 角色分布:")
            for role, count in roles.most_common():
                print(f"   • {role}: {count}条")
                
        if self.vector_metadata:
            types = Counter(m['type'] for m in self.vector_metadata)
            print(f"\n📋 向量类型分布:")
            for vtype, count in types.most_common():
                print(f"   • {vtype}: {count}个")
                
        print("\n" + "-" * 60)
    
    def print_detailed_statistics(self):
        """打印详细统计信息"""
        print("\n" + "=" * 60)
        print("📈 详细统计分析")
        print("=" * 60)
        
        # 时间分析
        if self.dialogs:
            timestamps = [datetime.fromisoformat(d['meta']['timestamp']) for d in self.dialogs]
            time_range = max(timestamps) - min(timestamps)
            print(f"\n⏰ 时间跨度:")
            print(f"   • 最早记录: {min(timestamps).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   • 最新记录: {max(timestamps).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   • 时间跨度: {time_range}")
            print(f"   • 日均记录: {len(self.dialogs) / max(time_range.days, 1):.2f}条")
        
        # 文本长度分析
        if self.paragraphs:
            lengths = [len(p['text']) for p in self.paragraphs]
            print(f"\n📝 段落文本长度:")
            print(f"   • 平均长度: {np.mean(lengths):.1f}字符")
            print(f"   • 最短: {min(lengths)}字符")
            print(f"   • 最长: {max(lengths)}字符")
            print(f"   • 中位数: {np.median(lengths):.1f}字符")
        
        if self.sentences:
            lengths = [len(s['text']) for s in self.sentences]
            print(f"\n📖 句子文本长度:")
            print(f"   • 平均长度: {np.mean(lengths):.1f}字符")
            print(f"   • 最短: {min(lengths)}字符")
            print(f"   • 最长: {max(lengths)}字符")
            print(f"   • 中位数: {np.median(lengths):.1f}字符")
        
        # 知识节点分析
        if self.knowledge_nodes:
            para_counts = [len(kn['paragraph_ids']) for kn in self.knowledge_nodes]
            print(f"\n🧠 知识节点分析:")
            print(f"   • 平均每节点段落数: {np.mean(para_counts):.1f}")
            print(f"   • 最多段落节点: {max(para_counts)}个段落")
            print(f"   • 最少段落节点: {min(para_counts)}个段落")
            
            # 关联关系分析
            total_relations = sum(len(kn['related_node_ids']) for kn in self.knowledge_nodes)
            print(f"   • 总关联关系数: {total_relations}")
            print(f"   • 平均每节点关联: {total_relations / len(self.knowledge_nodes):.1f}个")
        
        print("\n" + "-" * 60)
    
    # ==================== 可视化 ====================
    
    def visualize_all(self, save_dir="./analysis_output"):
        """生成所有可视化图表"""
        os.makedirs(save_dir, exist_ok=True)
        
        self._plot_distribution(save_dir)
        self._plot_knowledge_network(save_dir)
        self._plot_timeline(save_dir)
        self._plot_text_length_histogram(save_dir)
        self._plot_cluster_analysis(save_dir)
        
        print(f"\n📊 图表已保存到: {save_dir}")
    
    def _plot_distribution(self, save_dir):
        """Plot distribution charts"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Memory System Data Distribution', fontsize=16, fontweight='bold')

        # 1. Vector type distribution
        ax1 = axes[0, 0]
        types = Counter(m['type'] for m in self.vector_metadata)
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        bars = ax1.bar(types.keys(), types.values(), color=colors[:len(types)])
        ax1.set_title('Vector Type Distribution')
        ax1.set_ylabel('Count')
        for bar, val in zip(bars, types.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
                    str(val), ha='center', va='bottom')

        # 2. Role distribution
        ax2 = axes[0, 1]
        if self.dialogs:
            roles = Counter(d['meta']['role'] for d in self.dialogs)
            pie_colors = plt.cm.Set3(np.linspace(0, 1, len(roles)))
            wedges, texts, autotexts = ax2.pie(roles.values(), labels=roles.keys(),
                                                autopct='%1.1f%%', colors=pie_colors)
            ax2.set_title('Dialog Role Distribution')

        # 3. Knowledge node paragraph count distribution
        ax3 = axes[1, 0]
        if self.knowledge_nodes:
            para_counts = [len(kn['paragraph_ids']) for kn in self.knowledge_nodes]
            ax3.hist(para_counts, bins=range(min(para_counts), max(para_counts)+2),
                    edgecolor='black', alpha=0.7, color='#3498db')
            ax3.set_title('Knowledge Node Paragraph Count Distribution')
            ax3.set_xlabel('Paragraph Count')
            ax3.set_ylabel('Node Count')

        # 4. Text length boxplot (paragraphs)
        ax4 = axes[1, 1]
        if self.paragraphs:
            lengths = [len(p['text']) for p in self.paragraphs]
            ax4.boxplot(lengths, vert=True)
            ax4.set_title('Paragraph Text Length Boxplot')
            ax4.set_ylabel('Character Count')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(f"{save_dir}/distribution.png")

    def _plot_knowledge_network(self, save_dir):
        """Plot knowledge network graph"""
        if not self.knowledge_nodes:
            return

        fig, ax = plt.subplots(figsize=(12, 10))

        n_nodes = len(self.knowledge_nodes)
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        radius = 5
        positions = {kn['node_id']: (radius * np.cos(a), radius * np.sin(a))
                    for kn, a in zip(self.knowledge_nodes, angles)}

        for kn in self.knowledge_nodes:
            x, y = positions[kn['node_id']]
            para_count = len(kn['paragraph_ids'])
            size = 300 + para_count * 50
            ax.scatter(x, y, s=size, c='#3498db', alpha=0.7, zorder=3)
            ax.annotate(kn['node_id'], (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)

        for kn in self.knowledge_nodes:
            x1, y1 = positions[kn['node_id']]
            for related_id in kn['related_node_ids'][:3]:
                if related_id in positions:
                    x2, y2 = positions[related_id]
                    ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=1, zorder=1)

        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        ax.set_aspect('equal')
        ax.set_title('Knowledge Node Association Network', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/knowledge_network.png", dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(f"{save_dir}/knowledge_network.png")

    def _plot_timeline(self, save_dir):
        """Plot dialog timeline"""
        if not self.dialogs:
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        timestamps = [datetime.fromisoformat(d['meta']['timestamp']) for d in self.dialogs]
        dates = [t.date() for t in timestamps]
        date_counts = Counter(dates)

        sorted_dates = sorted(date_counts.keys())
        counts = [date_counts[d] for d in sorted_dates]

        bars = ax.bar(sorted_dates, counts, color='#2ecc71', alpha=0.7, width=0.8)
        ax.set_title('Dialog Timeline', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Dialogs')

        plt.xticks(rotation=45, ha='right')

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.1,
                    str(count), ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/timeline.png", dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(f"{save_dir}/timeline.png")

    def _plot_text_length_histogram(self, save_dir):
        """Plot text length histogram for paragraphs and sentences"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Text Length Distribution', fontsize=14, fontweight='bold')

        ax1 = axes[0]
        if self.paragraphs:
            lengths = [len(p['text']) for p in self.paragraphs]
            ax1.hist(lengths, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
            ax1.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
            ax1.legend()
        ax1.set_title('Paragraph Text Length Distribution')
        ax1.set_xlabel('Character Count')
        ax1.set_ylabel('Frequency')

        ax2 = axes[1]
        if self.sentences:
            lengths = [len(s['text']) for s in self.sentences]
            ax2.hist(lengths, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
            ax2.legend()
        ax2.set_title('Sentence Text Length Distribution')
        ax2.set_xlabel('Character Count')
        ax2.set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/text_length.png", dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(f"{save_dir}/text_length.png")

    def _plot_cluster_analysis(self, save_dir):
        """Plot cluster analysis of knowledge nodes"""
        if not self.knowledge_nodes:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        vectors = np.array([kn['center_vector'][:2] for kn in self.knowledge_nodes])

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors = vectors / norms

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.knowledge_nodes)))
        for i, kn in enumerate(self.knowledge_nodes):
            x, y = vectors[i]
            size = 200 + len(kn['paragraph_ids']) * 30
            ax.scatter(x, y, s=size, c=[colors[i]], alpha=0.6, label=kn['node_id'])

        for i, kn in enumerate(self.knowledge_nodes):
            x1, y1 = vectors[i]
            for related_id in kn['related_node_ids'][:2]:
                for j, other_kn in enumerate(self.knowledge_nodes):
                    if other_kn['node_id'] == related_id:
                        x2, y2 = vectors[j]
                        ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.2, linewidth=0.5)

        ax.set_title('Knowledge Node Cluster Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/cluster_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(f"{save_dir}/cluster_analysis.png")
        
        # ==================== 数据浏览 ====================
        
        def browse_paragraphs(self, page=1, per_page=10):
            """分页浏览段落"""
            start = (page - 1) * per_page
            end = start + per_page
            
            print(f"\n{'='*60}")
            print(f"📄 段落浏览 (第{page}页, 共{(len(self.paragraphs)-1)//per_page+1}页)")
            print("=" * 60)
            
            for i, p in enumerate(self.paragraphs[start:end], start=start+1):
                print(f"\n[{i}] ID: {p['id']}")
                print(f"    类型: {p['type']}")
                print(f"    父ID: {p.get('parent_tid', 'N/A')}")
                print(f"    时间: {p['timestamp']}")
                print(f"    文本: {p['text'][:100]}...")
                print("-" * 40)
            
            return len(self.paragraphs)
        
        def browse_sentences(self, page=1, per_page=10):
            """分页浏览句子"""
            start = (page - 1) * per_page
            end = start + per_page
            
            print(f"\n{'='*60}")
            print(f"📝 句子浏览 (第{page}页, 共{(len(self.sentences)-1)//per_page+1}页)")
            print("=" * 60)
            
            for i, s in enumerate(self.sentences[start:end], start=start+1):
                print(f"\n[{i}] ID: {s['id']}")
                print(f"    父段落: {s['parent_tid']}")
                print(f"    时间: {s['timestamp']}")
                print(f"    文本: {s['text']}")
                print("-" * 40)
            
            return len(self.sentences)
        
        def browse_knowledge_nodes(self):
            """浏览知识节点"""
            print(f"\n{'='*60}")
            print("🧠 知识节点浏览")
            print("=" * 60)
            
            for i, kn in enumerate(self.knowledge_nodes, 1):
                print(f"\n[{i}] 节点ID: {kn['node_id']}")
                print(f"    包含段落数: {len(kn['paragraph_ids'])}")
                print(f"    关联节点: {kn['related_node_ids']}")
                print(f"    段落IDs: {kn['paragraph_ids'][:5]}..." if len(kn['paragraph_ids']) > 5 else f"    段落IDs: {kn['paragraph_ids']}")
                print("-" * 40)
            
            return len(self.knowledge_nodes)
        
        def browse_dialogs(self, page=1, per_page=10, role=None):
            """分页浏览对话"""
            filtered = self.dialogs
            if role:
                filtered = [d for d in self.dialogs if d['meta']['role'] == role]
            
            start = (page - 1) * per_page
            end = start + per_page
            
            print(f"\n{'='*60}")
            print(f"💬 对话浏览 (第{page}页, 共{(len(filtered)-1)//per_page+1}页)")
            if role:
                print(f"    筛选角色: {role}")
            print("=" * 60)
            
            for i, d in enumerate(filtered[start:end], start=start+1):
                meta = d['meta']
                print(f"\n[{i}] TID: {meta['tid']}")
                print(f"    角色: {meta['role']}")
                print(f"    时间: {meta['timestamp']}")
                print(f"    内容: {d['text'][:200]}...")
                print("-" * 40)
            
            return len(filtered)
        
        # ==================== 数据修复 ====================
        
        def delete_paragraph(self, para_id):
            """删除段落及其相关句子"""
            # 找到段落元数据
            para_meta = next((p for p in self.paragraphs if p['id'] == para_id), None)
            if not para_meta:
                print(f"❌ 未找到段落: {para_id}")
                return False
            
            # 找到并删除相关句子
            related_sentences = [s for s in self.sentences if s['parent_tid'] == para_id]
            for s in related_sentences:
                self.sentences.remove(s)
                # 从vector_metadata中删除
                self.vector_metadata = [m for m in self.vector_metadata if m['id'] != s['id']]
            
            # 从vector_metadata中删除段落
            self.vector_metadata = [m for m in self.vector_metadata if m['id'] != para_id]
            self.paragraphs.remove(para_meta)
            
            # 从知识节点中删除关联
            for kn in self.knowledge_nodes:
                if para_id in kn['paragraph_ids']:
                    kn['paragraph_ids'].remove(para_id)
            
            # 保存更改
            self._save_changes()
            print(f"✅ 已删除段落: {para_id} 及相关 {len(related_sentences)} 个句子")
            return True
        
        def delete_sentence(self, sent_id):
            """删除句子"""
            sent_meta = next((s for s in self.sentences if s['id'] == sent_id), None)
            if not sent_meta:
                print(f"❌ 未找到句子: {sent_id}")
                return False
            
            self.sentences.remove(sent_meta)
            self.vector_metadata = [m for m in self.vector_metadata if m['id'] != sent_id]
            
            # 从知识节点中删除关联（需要重新计算）
            self._save_changes()
            print(f"✅ 已删除句子: {sent_id}")
            return True
        
        def delete_knowledge_node(self, node_id):
            """删除知识节点"""
            kn = next((k for k in self.knowledge_nodes if k['node_id'] == node_id), None)
            if not kn:
                print(f"❌ 未找到知识节点: {node_id}")
                return False
            
            self.knowledge_nodes.remove(kn)
            
            # 从其他节点的关联中删除
            for other_kn in self.knowledge_nodes:
                if node_id in other_kn['related_node_ids']:
                    other_kn['related_node_ids'].remove(node_id)
            
            self._save_changes()
            print(f"✅ 已删除知识节点: {node_id}")
            return True
        
        def edit_paragraph(self, para_id, new_text):
            """编辑段落文本"""
            para_meta = next((p for p in self.paragraphs if p['id'] == para_id), None)
            if not para_meta:
                print(f"❌ 未找到段落: {para_id}")
                return False
            
            old_text = para_meta['text']
            para_meta['text'] = new_text
            para_meta['timestamp'] = datetime.now().isoformat()
            
            # 更新vector_metadata
            for m in self.vector_metadata:
                if m['id'] == para_id:
                    m['text'] = new_text
                    m['timestamp'] = para_meta['timestamp']
                    break
            
            self._save_changes()
            print(f"✅ 已更新段落: {para_id}")
            print(f"   旧文本: {old_text[:50]}...")
            print(f"   新文本: {new_text[:50]}...")
            return True
        
        def merge_knowledge_nodes(self, node_ids, new_node_id=None):
            """合并多个知识节点"""
            if len(node_ids) < 2:
                print("❌ 至少需要2个节点才能合并")
                return False
            
            # 验证所有节点存在
            nodes_to_merge = []
            for nid in node_ids:
                kn = next((k for k in self.knowledge_nodes if k['node_id'] == nid), None)
                if kn:
                    nodes_to_merge.append(kn)
                else:
                    print(f"❌ 未找到节点: {nid}")
                    return False
            
            # 创建新节点
            if new_node_id is None:
                new_node_id = f"kno_merged_{int(datetime.now().timestamp())}"
            
            # 合并段落ID
            merged_paragraph_ids = []
            for kn in nodes_to_merge:
                merged_paragraph_ids.extend(kn['paragraph_ids'])
            merged_paragraph_ids = list(set(merged_paragraph_ids))  # 去重
            
            # 计算新的中心向量
            all_vectors = []
            for pid in merged_paragraph_ids:
                para_meta = next((p for p in self.paragraphs if p['id'] == pid), None)
                if para_meta:
                    # 这里需要从vector_index获取实际向量
                    all_vectors.append(np.array(para_meta.get('vector', [0]*VECTOR_DIM)))
            
            if all_vectors:
                new_center = np.mean(all_vectors, axis=0)
                new_center = new_center / (np.linalg.norm(new_center) + 1e-8)
            else:
                new_center = np.zeros(VECTOR_DIM)
            
            # 创建新节点
            new_node = {
                'node_id': new_node_id,
                'center_vector': new_center.tolist(),
                'paragraph_ids': merged_paragraph_ids,
                'related_node_ids': []
            }
            self.knowledge_nodes.append(new_node)
            
            # 删除旧节点
            for kn in nodes_to_merge:
                self.knowledge_nodes.remove(kn)
                # 从其他节点的关联中删除
                for other_kn in self.knowledge_nodes:
                    if kn['node_id'] in other_kn['related_node_ids']:
                        other_kn['related_node_ids'].remove(kn['node_id'])
            
            # 建立新节点的关联
            self._update_node_relations()
            
            self._save_changes()
            print(f"✅ 已合并 {len(node_ids)} 个节点为: {new_node_id}")
            print(f"   合并后段落数: {len(merged_paragraph_ids)}")
            return True
        
        def _update_node_relations(self):
            """更新知识节点间的关联关系"""
            nodes = self.knowledge_nodes
            for i in range(len(nodes)):
                sims = []
                for j in range(len(nodes)):
                    if i != j:
                        v1 = np.array(nodes[i]['center_vector'])
                        v2 = np.array(nodes[j]['center_vector'])
                        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                        sims.append((j, sim))
                sims.sort(key=lambda x: x[1], reverse=True)
                nodes[i]['related_node_ids'] = [nodes[j]['node_id'] for j, _ in sims[:3]]
        
        def _save_changes(self):
            """保存所有更改"""
            # 保存vector_metadata
            with open(f"{VECTOR_DB_DIR}/metadata.json", 'w', encoding='utf-8') as f:
                json.dump(self.vector_metadata, f, ensure_ascii=False, indent=2)
            
            # 保存knowledge_data
            kg_data = {}
            for kn in self.knowledge_nodes:
                kg_data[kn['node_id']] = {
                    'center_vector': kn['center_vector'],
                    'paragraph_ids': kn['paragraph_ids'],
                    'related_node_ids': kn['related_node_ids']
                }
            with open(f"{KNOWLEDGE_DIR}/knowledge_graph.json", 'w', encoding='utf-8') as f:
                json.dump(kg_data, f, ensure_ascii=False, indent=2)
            
            # 重新构建内存结构
            self._build_memory_structures()
        
        # ==================== 搜索功能 ====================
        
        def search_content(self, keyword, search_in=['paragraphs', 'sentences', 'dialogs']):
            """搜索关键词"""
            results = {'paragraphs': [], 'sentences': [], 'dialogs': []}
            
            if 'paragraphs' in search_in:
                for p in self.paragraphs:
                    if keyword.lower() in p['text'].lower():
                        results['paragraphs'].append(p)
            
            if 'sentences' in search_in:
                for s in self.sentences:
                    if keyword.lower() in s['text'].lower():
                        results['sentences'].append(s)
            
            if 'dialogs' in search_in:
                for d in self.dialogs:
                    if keyword.lower() in d['text'].lower():
                        results['dialogs'].append(d)
            
            return results
        
        def search_by_date_range(self, start_date, end_date):
            """按日期范围搜索"""
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            
            filtered = []
            for d in self.dialogs:
                ts = datetime.fromisoformat(d['meta']['timestamp'])
                if start <= ts <= end:
                    filtered.append(d)
            
            return filtered
        
        # ==================== 导出功能 ====================
        
        def export_to_json(self, filepath):
            """导出所有数据到JSON文件"""
            export_data = {
                'export_time': datetime.now().isoformat(),
                'summary': {
                    'total_vectors': len(self.vector_metadata),
                    'total_paragraphs': len(self.paragraphs),
                    'total_sentences': len(self.sentences),
                    'total_knowledge_nodes': len(self.knowledge_nodes),
                    'total_dialogs': len(self.dialogs)
                },
                'paragraphs': self.paragraphs,
                'sentences': self.sentences,
                'knowledge_nodes': self.knowledge_nodes,
                'dialogs': [{
                    'meta': d['meta'],
                    'text': d['text']
                } for d in self.dialogs]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 数据已导出到: {filepath}")
            return export_data
        
        def generate_report(self, filepath):
            """生成分析报告"""
            report = []
            report.append("# 记忆系统分析报告")
            report.append(f"\n生成时间: {datetime.now().isoformat()}")
            report.append("\n## 1. 概览")
            report.append(f"- 总向量数: {len(self.vector_metadata)}")
            report.append(f"- 句子级向量: {len(self.sentences)}")
            report.append(f"- 段落级向量: {len(self.paragraphs)}")
            report.append(f"- 知识节点: {len(self.knowledge_nodes)}")
            report.append(f"- 对话记录: {len(self.dialogs)}")
            
            if self.dialogs:
                report.append("\n## 2. 角色分布")
                roles = Counter(d['meta']['role'] for d in self.dialogs)
                for role, count in roles.most_common():
                    report.append(f"- {role}: {count}条")
            
            if self.vector_metadata:
                report.append("\n## 3. 向量类型分布")
                types = Counter(m['type'] for m in self.vector_metadata)
                for vtype, count in types.most_common():
                    report.append(f"- {vtype}: {count}个")
            
            if self.knowledge_nodes:
                report.append("\n## 4. 知识节点分析")
                para_counts = [len(kn['paragraph_ids']) for kn in self.knowledge_nodes]
                report.append(f"- 平均每节点段落数: {np.mean(para_counts):.1f}")
                report.append(f"- 总关联关系数: {sum(len(kn['related_node_ids']) for kn in self.knowledge_nodes)}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            print(f"✅ 报告已生成: {filepath}")
            return '\n'.join(report)


def main():
    """主函数 - 交互式分析界面"""
    print("=" * 60)
    print("🔍 记忆系统分析工具")
    print("=" * 60)
    
    analyzer = MemoryAnalyzer()
    analyzer.load_from_files()
    
    while True:
        print("\n" + "-" * 40)
        print("请选择操作:")
        print("1. 📊 查看概览")
        print("2. 📈 详细统计")
        print("3. 📄 浏览段落")
        print("4. 📝 浏览句子")
        print("5. 🧠 浏览知识节点")
        print("6. 💬 浏览对话")
        print("7. 🔍 搜索内容")
        print("8. 📊 生成可视化图表")
        print("9. 🗑️  删除段落")
        print("10. ✏️  编辑段落")
        print("11. 🔗 合并知识节点")
        print("12. 📤 导出数据")
        print("13. 📝 生成报告")
        print("0. 退出")
        print("-" * 40)
        
        choice = input("请输入选项: ").strip()
        
        if choice == '1':
            analyzer.print_overview()
        elif choice == '2':
            analyzer.print_detailed_statistics()
        elif choice == '3':
            page = int(input("输入页码 (默认1): ") or "1")
            per_page = int(input("每页数量 (默认10): ") or "10")
            analyzer.browse_paragraphs(page, per_page)
        elif choice == '4':
            page = int(input("输入页码 (默认1): ") or "1")
            per_page = int(input("每页数量 (默认10): ") or "10")
            analyzer.browse_sentences(page, per_page)
        elif choice == '5':
            analyzer.browse_knowledge_nodes()
        elif choice == '6':
            page = int(input("输入页码 (默认1): ") or "1")
            per_page = int(input("每页数量 (默认10): ") or "10")
            role = input("筛选角色 (可选): ").strip() or None
            analyzer.browse_dialogs(page, per_page, role)
        elif choice == '7':
            keyword = input("输入搜索关键词: ").strip()
            if keyword:
                results = analyzer.search_content(keyword)
                print(f"\n找到 {len(results['paragraphs'])} 个段落, "
                      f"{len(results['sentences'])} 个句子, "
                      f"{len(results['dialogs'])} 条对话")
        elif choice == '8':
            save_dir = input("输出目录 (默认./analysis_output): ").strip() or "./analysis_output"
            analyzer.visualize_all(save_dir)
        elif choice == '9':
            para_id = input("输入要删除的段落ID: ").strip()
            confirm = input(f"确认删除 {para_id}? (y/n): ").strip().lower()
            if confirm == 'y':
                analyzer.delete_paragraph(para_id)
        elif choice == '10':
            para_id = input("输入要编辑的段落ID: ").strip()
            new_text = input("输入新文本: ").strip()
            if new_text:
                analyzer.edit_paragraph(para_id, new_text)
        elif choice == '11':
            node_ids = input("输入要合并的节点ID (逗号分隔): ").strip().split(',')
            node_ids = [n.strip() for n in node_ids if n.strip()]
            analyzer.merge_knowledge_nodes(node_ids)
        elif choice == '12':
            filepath = input("输出文件路径 (默认./export.json): ").strip() or "./export.json"
            analyzer.export_to_json(filepath)
        elif choice == '13':
            filepath = input("报告文件路径 (默认./report.md): ").strip() or "./report.md"
            analyzer.generate_report(filepath)
        elif choice == '0':
            print("再见! 👋")
            break
        else:
            print("无效选项，请重试")


if __name__ == "__main__":
    main()