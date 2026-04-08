import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Callable
import argparse
from tqdm import tqdm
import requests
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OllamaClient:
    """Ollama本地模型客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
    
    def generate_response(self, model: str, prompt: str, temperature: float = 0.1) -> str:
        """调用Ollama模型生成响应"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9,
                        "top_k": 40
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API调用失败: {response.status_code} - {response.text}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama请求异常: {e}")
            return ""
        except Exception as e:
            logger.error(f"Ollama处理异常: {e}")
            return ""
    
    def generate_response_with_retry(self, model: str, prompt: str, max_retries: int = 3) -> str:
        """带重试的模型调用"""
        for attempt in range(max_retries):
            try:
                response = self.generate_response(model, prompt)
                if response and len(response.strip()) > 0:
                    return response
                else:
                    logger.warning(f"第{attempt+1}次尝试得到空响应，正在重试...")
                    time.sleep(2)
            except Exception as e:
                logger.warning(f"第{attempt+1}次尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        logger.error(f"所有{max_retries}次尝试均失败")
        return "评估失败：模型无响应"
    
    def evaluate_answer_correctness(self, model: str, question: str, 
                                 generated_answer: str, ground_truth: str) -> Tuple[float, Dict[str, Any]]:
        """使用Ollama评估答案正确性，返回分数和详细评估信息"""
        prompt_template = """
请评估以下问答对的正确性。请严格根据标准答案判断生成答案的准确性。

问题：{question}

生成的答案：{generated_answer}

标准答案：{ground_truth}

请从以下选项中选择最合适的评价，并提供简要的理由说明：

A. 完全正确（生成答案包含所有关键信息且准确无误）
B. 部分正确（生成答案包含部分关键信息但有不准确或缺失）
C. 基本错误（生成答案与标准答案相关性低）
D. 完全错误（生成答案与问题无关或严重错误）

请返回JSON格式：
{{
    "choice": "A/B/C/D",
    "reason": "选择这个选项的理由说明"
}}
        """
        
        prompt = prompt_template.format(
            question=question,
            generated_answer=generated_answer,
            ground_truth=ground_truth
        )
        
        response = self.generate_response_with_retry(model, prompt)
        
        # 解析评估结果
        score, evaluation_details = self._parse_evaluation_response(response)
        evaluation_details["raw_response"] = response
        
        return score, evaluation_details
    
    def _parse_evaluation_response(self, response: str) -> Tuple[float, Dict[str, Any]]:
        """解析评估响应为分数和详细信息"""
        response_clean = response.strip()
        
        # 默认评估详情
        evaluation_details = {
            "choice": "未知",
            "reason": "无法解析评估响应",
            "parsing_success": False
        }
        
        try:
            # 尝试解析JSON格式的响应
            if response_clean.startswith("{"):
                response_json = json.loads(response_clean)
                choice = response_json.get("choice", "").upper()
                reason = response_json.get("reason", "未提供理由")
                
                evaluation_details.update({
                    "choice": choice,
                    "reason": reason,
                    "parsing_success": True
                })
                
                # 根据选项分配分数
                if 'A' in choice or '完全正确' in response_clean:
                    return 1.0, evaluation_details
                elif 'B' in choice or '部分正确' in response_clean:
                    return 0.7, evaluation_details
                elif 'C' in choice or '基本错误' in response_clean:
                    return 0.3, evaluation_details
                elif 'D' in choice or '完全错误' in response_clean:
                    return 0.0, evaluation_details
            else:
                # 非JSON响应，尝试提取选项
                for char in response_clean:
                    if char in "ABCD":
                        evaluation_details["choice"] = char
                        evaluation_details["reason"] = f"从文本中提取的选项: {char}"
                        evaluation_details["parsing_success"] = True
                        
                        if char == 'A':
                            return 1.0, evaluation_details
                        elif char == 'B':
                            return 0.7, evaluation_details
                        elif char == 'C':
                            return 0.3, evaluation_details
                        elif char == 'D':
                            return 0.0, evaluation_details
                
                # 尝试从文本中识别
                if '完全正确' in response_clean or 'A' in response_clean.upper():
                    evaluation_details.update({"choice": "A", "reason": "文本匹配: 完全正确", "parsing_success": True})
                    return 1.0, evaluation_details
                elif '部分正确' in response_clean or 'B' in response_clean.upper():
                    evaluation_details.update({"choice": "B", "reason": "文本匹配: 部分正确", "parsing_success": True})
                    return 0.7, evaluation_details
                elif '基本错误' in response_clean or 'C' in response_clean.upper():
                    evaluation_details.update({"choice": "C", "reason": "文本匹配: 基本错误", "parsing_success": True})
                    return 0.3, evaluation_details
                elif '完全错误' in response_clean or 'D' in response_clean.upper():
                    evaluation_details.update({"choice": "D", "reason": "文本匹配: 完全错误", "parsing_success": True})
                    return 0.0, evaluation_details
                
        except json.JSONDecodeError:
            logger.warning(f"JSON解析失败: {response_clean[:100]}...")
        
        # 无法解析时使用保守分数
        logger.warning(f"无法解析评估响应: {response_clean[:100]}...")
        return 0.5, evaluation_details

class LongMemEvalEvaluator:
    """LongMemEval基准测评器（Ollama版本）- 与 eval.py 相同逻辑，可注入不同 memory_manager"""
    
    def __init__(self, memory_manager, config_path: str = "eval_config.json"):
        self.memory_manager = memory_manager
        self.config = self._load_config(config_path)
        self.ollama_client = OllamaClient(self.config.get("ollama_base_url", "http://localhost:11434"))
        self.results = {}
        self.detailed_results = []
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载评估配置"""
        default_config = {
            "benchmark_path": "data/longmemeval",
            "output_dir": "results_new",
            "evaluation_llm": "gpt-oss:20b",
            "ollama_base_url": "http://localhost:11434",
            "max_retrieved_items": 10,
            "timeout_seconds": 120,
            "question_types": [
                "single-session-user", "single-session-assistant", 
                "single-session-preference", "multi-session", 
                "knowledge-update", "temporal-reasoning", "absention"
            ],
            "metrics": ["accuracy", "recall@k", "ndcg@k", "f1_score"],
            "save_detailed_results": True,
            "detailed_results_threshold": 50,
            "tune_train_ratio": 0.7,
            "tune_seed": 42,
            "tune_threshold_candidates": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            "tune_reward_recall_weight": 0.5,
            "tune_reward_ndcg_weight": 0.5,
        }
        
        # 支持 config/ 前缀与当前目录；记录实际加载路径，便于调参后写回
        resolved_path = config_path
        for path in [config_path, os.path.join("config", config_path)]:
            if os.path.exists(path):
                resolved_path = path
                with open(path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                break
        default_config["_config_file_path"] = resolved_path
        return default_config
    
    def load_benchmark_data(self, dataset_type: str = "oracle") -> List[Dict[str, Any]]:
        """加载LongMemEval基准数据（与 eval.py 相同）"""
        dataset_path = os.path.join(
            self.config["benchmark_path"], 
            f"longmemeval_{dataset_type}.json"
        )
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"基准数据集不存在: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            instances = json.load(f)
        
        converted_instances = []
        
        for i, instance in enumerate(instances):
            converted_instance = {
                "instance_id": instance.get("question_id", f"instance_{i}"),
                "question_type": instance.get("question_type", "unknown"),
                "question": instance.get("question", ""),
                "answer": instance.get("answer", ""),
                "sessions": [],
                "evidence_sessions": instance.get("answer_session_ids", [])
            }
            
            if "haystack_sessions" in instance:
                haystack_session_ids = instance.get("haystack_session_ids", [])
                for session_idx, session in enumerate(instance["haystack_sessions"]):
                    haystack_sid = haystack_session_ids[session_idx] if session_idx < len(haystack_session_ids) else None
                    for turn in session:
                        converted_instance["sessions"].append({
                            "role": turn["role"],
                            "text": turn["content"],
                            "session_id": session_idx,
                            "turn_id": f"session_{session_idx}_turn_{len(converted_instance['sessions'])}",
                            "haystack_session_id": haystack_sid
                        })
            else:
                if "dialog_history" in instance:
                    for turn in instance["dialog_history"]:
                        converted_instance["sessions"].append({
                            "role": turn["role"],
                            "text": turn["content"]
                        })
            
            converted_instances.append(converted_instance)
        
        logger.info(f"加载 {len(converted_instances)} 个评估实例，共 {sum(len(inst['sessions']) for inst in converted_instances)} 条对话记录")
        return converted_instances
    
    def split_benchmark_data(
        self,
        instances: List[Dict[str, Any]],
        train_ratio: float = 0.7,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        划分训练集与验证集，用于异步调参。
        数据总量 = 传入的 instances 长度（通常来自 load_benchmark_data(dataset_type) 全量）。
        train_ratio/seed 可由 run_threshold_tuning 的入参或 config 传入。
        """
        rng = np.random.default_rng(seed)
        n = len(instances)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_train = max(1, int(n * train_ratio))
        train_idx, val_idx = idx[:n_train], idx[n_train:]
        train_instances = [instances[i] for i in train_idx]
        val_instances = [instances[i] for i in val_idx]
        return train_instances, val_instances
    
    def run_retrieval_only_eval(
        self,
        instances: List[Dict[str, Any]],
        retrieval_params: Dict[str, float],
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> Tuple[float, float]:
        """
        仅做检索评估（不调 LLM），用于调参。返回 (平均 recall@k, 平均 ndcg@k)。
        progress_callback: 每跑完一条实例后调用一次，用于外层进度条按实例推进。
        """
        from memory_manager import VectorMemoryManager
        old_mgr = self.memory_manager
        self.memory_manager = VectorMemoryManager(retrieval_params=retrieval_params)
        recalls, ndcgs = [], []
        prev_tuning = getattr(self, "_in_tuning", False)
        self._in_tuning = True
        try:
            for inst in instances:
                self.setup_evaluation_environment(inst)
                qt = inst.get("question_type", "")
                is_temporal = getattr(self.memory_manager, "is_temporal_task", lambda _: False)(qt)
                retrieved = self.memory_manager.search(
                    inst["question"],
                    top_k=self.config["max_retrieved_items"],
                    is_temporal_task=is_temporal,
                )
                rm = self._evaluate_retrieval_quality(retrieved, inst)
                recalls.append(rm["recall@k"])
                ndcgs.append(rm["ndcg@k"])
                if progress_callback:
                    progress_callback()
            return (float(np.mean(recalls)), float(np.mean(ndcgs)))
        finally:
            self.memory_manager = old_mgr
            self._in_tuning = prev_tuning
    
    def run_threshold_tuning(
        self,
        dataset_type: str = "oracle",
        train_ratio: Optional[float] = None,
        tune_seed: Optional[int] = None,
        max_workers: int = 4,
    ) -> Dict[str, float]:
        """
        异步在训练集上搜索最优“多少算相关”阈值，以 recall@k 与 ndcg@k 为奖励，写入 config。
        数据总量：load_benchmark_data(dataset_type) 全量；划分比例 train_ratio 来自入参或 config["tune_train_ratio"]。
        """
        instances = self.load_benchmark_data(dataset_type)
        if len(instances) < 2:
            logger.warning("实例过少，跳过阈值调参")
            return {}
        train_ratio = train_ratio if train_ratio is not None else self.config.get("tune_train_ratio", 0.7)
        seed = tune_seed if tune_seed is not None else self.config.get("tune_seed", 42)
        train_instances, val_instances = self.split_benchmark_data(
            instances, train_ratio=train_ratio, seed=seed
        )
        logger.info(
            "调参数据划分: 总量=%d, 训练集=%d (%.0f%%), 验证集=%d (train_ratio=%.2f, seed=%d)",
            len(instances), len(train_instances), train_ratio * 100, len(val_instances), train_ratio, seed,
        )
        candidates = self.config.get("tune_threshold_candidates", [0.15, 0.25, 0.35, 0.4, 0.45, 0.55, 0.6])
        w_recall = self.config.get("tune_reward_recall_weight", 0.5)
        w_ndcg = self.config.get("tune_reward_ndcg_weight", 0.5)
        config_file_path = self.config.get("_config_file_path", "eval_config.json")
        total_steps = len(candidates) * len(train_instances)
        pbar_lock = threading.Lock()
        pbar = tqdm(total=total_steps, desc="阈值调参(按实例)", unit="实例")

        def progress_cb():
            with pbar_lock:
                pbar.update(1)

        def eval_one(th: float) -> Tuple[float, float, float]:
            params = {"min_summary_similarity": th, "knowledge_min_score": th}
            recall, ndcg = self.run_retrieval_only_eval(
                train_instances, params, progress_callback=progress_cb
            )
            return (th, recall, ndcg)

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(eval_one, th): th for th in candidates}
            for fut in as_completed(futures):
                try:
                    th, recall, ndcg = fut.result()
                    results.append((th, recall, ndcg))
                except Exception as e:
                    logger.warning("某阈值评估失败: %s", e)
        pbar.close()
        
        if not results:
            return {}
        best = max(results, key=lambda x: x[1] * w_recall + x[2] * w_ndcg)
        best_th, best_recall, best_ndcg = best
        self.config["min_summary_similarity"] = best_th
        self.config["knowledge_min_score"] = best_th
        # 写回 eval_config.json，只更新检索相关键，保留原有内容
        try:
            current = {}
            if os.path.exists(config_file_path):
                with open(config_file_path, "r", encoding="utf-8") as f:
                    current = json.load(f)
            current["min_summary_similarity"] = best_th
            current["knowledge_min_score"] = best_th
            os.makedirs(os.path.dirname(config_file_path) or ".", exist_ok=True)
            with open(config_file_path, "w", encoding="utf-8") as f:
                json.dump(current, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("写回 eval_config 失败: %s", e)
        logger.info(
            "阈值调参完成: 最佳 min_summary_similarity=knowledge_min_score=%.3f, train recall=%.3f ndcg=%.3f, 已写入 %s",
            best_th, best_recall, best_ndcg, config_file_path,
        )
        return {"min_summary_similarity": best_th, "knowledge_min_score": best_th}
    
    def setup_evaluation_environment(self, instance: Dict[str, Any]) -> None:
        """设置评估环境，加载对话历史，并建立 answer_session_ids -> tid 映射"""
        self._reset_memory()
        
        session_tid_map = defaultdict(list)
        sessions = instance["sessions"]
        _disable_session_bar = getattr(self, "_in_tuning", False)
        for session in tqdm(sessions, desc="加载会话历史", leave=False, disable=_disable_session_bar):
            tid = self.memory_manager.add_dialog(
                role=session["role"],
                text=session["text"]
            )
            haystack_sid = session.get("haystack_session_id")
            if haystack_sid is not None:
                session_tid_map[haystack_sid].append(tid)
        
        answer_session_ids = instance.get("evidence_sessions", [])
        if answer_session_ids and session_tid_map:
            evidence_tids = []
            for aid in answer_session_ids:
                evidence_tids.extend(session_tid_map.get(aid, []))
            instance["evidence_sessions"] = list(set(evidence_tids))
    
    def _reset_memory(self) -> None:
        """重置记忆系统（兼容 memory_manager 的 reset）"""
        if hasattr(self.memory_manager, 'reset'):
            self.memory_manager.reset()
        else:
            if hasattr(self.memory_manager, 'vector_index'):
                import faiss
                from config import VECTOR_DIM
                self.memory_manager.vector_index = faiss.IndexFlatIP(VECTOR_DIM)
            if hasattr(self.memory_manager, 'vector_metadata'):
                self.memory_manager.vector_metadata = []
            talk_file = getattr(self.memory_manager, 'talk_file', 'data/talk.txt')
            if os.path.exists(talk_file):
                os.remove(talk_file)
            os.makedirs(os.path.dirname(talk_file), exist_ok=True)
            open(talk_file, 'w').close()
    
    def evaluate_single_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个实例"""
        try:
            self.setup_evaluation_environment(instance)
            
            query = instance["question"]
            question_type = instance.get("question_type", "")
            is_temporal = self.memory_manager.is_temporal_task(question_type) if hasattr(self.memory_manager, "is_temporal_task") else False
            retrieved_results = self.memory_manager.search(
                query,
                top_k=self.config["max_retrieved_items"],
                is_temporal_task=is_temporal,
            )
            
            retrieval_metrics = self._evaluate_retrieval_quality(
                retrieved_results, instance
            )
            
            qa_metrics, qa_details = self._evaluate_qa_quality(
                retrieved_results, instance
            )
            
            detailed_result = {
                "instance_id": instance["instance_id"],
                "question_type": instance["question_type"],
                "question": instance["question"],
                "ground_truth": instance["answer"],
                "retrieved_items": len(retrieved_results),
                "retrieved_results": self._format_retrieved_results_for_storage(retrieved_results),
                "generated_answer": qa_metrics.get("generated_answer", ""),
                "evaluation_details": qa_details,
                "retrieval_metrics": retrieval_metrics,
                "qa_metrics": qa_metrics,
                "timestamp": datetime.now().isoformat()
            }
            self.detailed_results.append(detailed_result)
            
            return {
                "instance_id": instance["instance_id"],
                "question_type": instance["question_type"],
                "retrieval_metrics": retrieval_metrics,
                "qa_metrics": qa_metrics,
                "retrieved_items": len(retrieved_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"评估实例 {instance.get('instance_id', 'unknown')} 失败: {e}")
            self.detailed_results.append({
                "instance_id": instance.get("instance_id", "unknown"),
                "question": instance.get("question", ""),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return {
                "instance_id": instance.get("instance_id", "unknown"),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _format_retrieved_results_for_storage(self, retrieved_results: List[Any]) -> List[Dict[str, Any]]:
        """格式化检索结果以便存储（兼容 knowledge_paragraph / summary_paragraph 等）"""
        formatted_results = []
        for i, result in enumerate(retrieved_results):
            raw_text = (
                result.get('full_dialog')
                or result.get('full_text')
                or result.get('text', '')
            )
            result_type = result.get('type', '')
            if not result_type and 'cluster_id' in result:
                result_type = 'knowledge_item'
            formatted_result = {
                "rank": i + 1,
                "text": raw_text[:1000] if raw_text else '',
                "score": float(result.get('score', 0)) if hasattr(result.get('score', 0), '__float__') else 0,
                "tid": result.get('tid', ''),
                "type": result_type,
                "timestamp": result.get('timestamp', ''),
                "dialog_timestamp": result.get('dialog_timestamp', ''),
                "role": result.get('role', 'unknown'),
                "session_id": result.get('session_id', ''),
                "turn_id": result.get('turn_id', '')
            }
            if result.get('cluster_id'):
                formatted_result["cluster_id"] = result['cluster_id']
            formatted_results.append(formatted_result)
        return formatted_results
    
    def _evaluate_retrieval_quality(self, retrieved_results: List[Any], 
                                   instance: Dict[str, Any]) -> Dict[str, float]:
        """评估检索质量"""
        ground_truth_evidence = instance.get("evidence_sessions", [])
        if not ground_truth_evidence:
            return {"recall@k": 0.0, "precision@k": 0.0, "ndcg@k": 0.0}
        recall = self._calculate_recall(retrieved_results, ground_truth_evidence)
        precision = self._calculate_precision(retrieved_results, ground_truth_evidence)
        ndcg = self._calculate_ndcg(retrieved_results, ground_truth_evidence)
        return {"recall@k": recall, "precision@k": precision, "ndcg@k": ndcg}
    
    def _evaluate_qa_quality(self, retrieved_results: List[Any], 
                           instance: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """评估问答质量 - 使用Ollama模型"""
        generated_answer = self._generate_answer_with_ollama(instance, retrieved_results)
        ground_truth = instance["answer"]
        correctness_score, evaluation_details = self.ollama_client.evaluate_answer_correctness(
            model=self.config["evaluation_llm"],
            question=instance["question"],
            generated_answer=generated_answer,
            ground_truth=ground_truth
        )
        qa_metrics = {
            "correctness": correctness_score,
            "answer_length": len(generated_answer),
            "generated_answer": generated_answer,
            "ground_truth": ground_truth
        }
        return qa_metrics, evaluation_details
    
    def _calculate_recall(self, retrieved: List[Any], ground_truth: List[str]) -> float:
        if not ground_truth:
            return 0.0
        retrieved_ids = {str(item.get('tid', '')) for item in retrieved}
        ground_truth_ids = set(ground_truth)
        intersection = retrieved_ids.intersection(ground_truth_ids)
        return len(intersection) / len(ground_truth_ids) if ground_truth_ids else 0.0
    
    def _calculate_precision(self, retrieved: List[Any], ground_truth: List[str]) -> float:
        if not retrieved:
            return 0.0
        retrieved_ids = {str(item.get('tid', '')) for item in retrieved}
        ground_truth_ids = set(ground_truth)
        intersection = retrieved_ids.intersection(ground_truth_ids)
        return len(intersection) / len(retrieved_ids) if retrieved_ids else 0.0
    
    def _calculate_ndcg(self, retrieved: List[Any], ground_truth: List[str]) -> float:
        relevance_scores = []
        for item in retrieved:
            tid = str(item.get('tid', ''))
            score = 1.0 if tid in ground_truth else 0.0
            relevance_scores.append(score)
        if not relevance_scores:
            return 0.0
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
        ideal_relevance = [1.0] * min(len(ground_truth), len(relevance_scores))
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        return dcg / idcg if idcg > 0 else 0.0
    
    def _generate_answer_with_ollama(self, instance: Dict[str, Any], retrieved_results: List[Any]) -> str:
        """根据 instance 的 question_type 选提示模板，将记忆检索结果与问题直接组合后调用模型。"""
        question = instance.get("question", "")
        question_type = (instance.get("question_type") or "").strip()
        if hasattr(self.memory_manager, 'format_context_for_llm'):
            context = self.memory_manager.format_context_for_llm(retrieved_results)
        else:
            context = self._format_retrieved_context_for_prompt(retrieved_results, question_type)
        prompt = self._build_prompt(question_type, context, question)
        return self.ollama_client.generate_response_with_retry(
            model=self.config["evaluation_llm"],
            prompt=prompt
        )

    def _format_retrieved_context_for_prompt(self, retrieved_results: List[Any], question_type: str = None) -> str:
        """将检索结果格式化为可直接与问题组合的文本（记忆检索结果部分）。"""
        if not retrieved_results:
            return "（无相关记忆检索结果）"
        
        parts = []
        for i, result in enumerate(retrieved_results):
            text = (
                result.get('full_dialog')
                or result.get('full_text')
                or result.get('text', '')
            )
            text = (text or "").strip()
            if not text:
                continue
            
            # 对于 single-session-assistant 问题，优化显示助手内容
            if question_type and question_type.strip().lower() == "single-session-assistant":
                # 检查文本中是否包含助手的内容
                if "assistant:" in text.lower():
                    # 提取所有对话行
                    lines = text.split('\n')
                    assistant_lines = []
                    other_lines = []
                    
                    for line in lines:
                        line_lower = line.lower()
                        if line_lower.startswith("assistant:"):
                            # 助手的内容，突出显示
                            assistant_lines.append(f"👉 {line}")
                        elif line_lower.startswith("user:"):
                            # 用户的内容，保留但可以简化
                            other_lines.append(f"   {line}")
                        else:
                            # 其他内容
                            other_lines.append(line)
                    
                    # 如果找到助手内容，优先显示助手内容
                    if assistant_lines:
                        # 组合内容：先显示助手内容，然后是其他内容
                        combined_lines = assistant_lines
                        if other_lines:
                            combined_lines.append("\n【其他相关对话】")
                            combined_lines.extend(other_lines)
                        text = "\n".join(combined_lines)
            
            ts = result.get('dialog_timestamp') or result.get('timestamp', '')
            if isinstance(ts, datetime):
                ts = ts.isoformat()
            label = f"[{i+1}]"
            if ts:
                parts.append(f"{label} [时间: {ts}]\n{text}")
            else:
                parts.append(f"{label}\n{text}")
        
        # 对于 single-session-assistant 问题，添加总体说明
        if question_type and question_type.strip().lower() == "single-session-assistant":
            if parts:
                return "注意：以下检索结果中包含了助手（assistant）的历史回答（以👉标记）。请特别关注助手提供的信息来回答问题。\n\n" + "\n\n".join(parts)
        
        return "\n\n".join(parts) if parts else "（无相关记忆检索结果）"

    def _build_prompt(self, question_type: str, context: str, question: str) -> str:
        """根据问题类型（统一接口，非关键词）选择说明语，并将记忆检索结果与问题直接组合。"""
        instruction = self._get_instruction_by_question_type(question_type)
        return f"""{instruction}

【记忆检索结果】
{context}

【问题】
{question}

请基于上述记忆检索结果回答问题。若检索结果不足以回答，请说明。"""

    def _get_instruction_by_question_type(self, question_type: str) -> str:
        """根据 question_type 返回对应说明语（统一接口，弃用关键词判断）。"""
        qt = (question_type or "").strip().lower()
        if qt == "temporal-reasoning":
            return "你是一个擅长时间推理的助手。检索结果已按时间顺序排列，请根据事件先后顺序作答。"
        if qt == "single-session-preference":
            return "你是一个了解用户偏好的助手。请根据对话历史总结用户偏好并作答。"
        if qt in ("multi-session", "knowledge-update"):
            return "你是一个跨会话记忆助手。请综合多轮对话信息作答。"
        if qt == "single-session-user":
            return "你是一个对话记忆助手。请根据检索到的对话内容准确作答，特别关注用户（user）的历史对话。"
        if qt == "single-session-assistant":
            return "你是一个对话记忆助手。请根据检索到的对话内容准确作答，特别关注助手（assistant）的历史回答。问题要求回忆助手之前提供的信息，请仔细检查助手的历史回答。"
        return "你是一个专业助手。请严格基于上述记忆检索结果作答。"
    
    def run_evaluation(
        self,
        dataset_type: str = "oracle",
        sample_size: int = None,
        instances: Optional[List[Dict[str, Any]]] = None,
        sample_seed: Optional[int] = None,
        run_tag: Optional[str] = None,
        skip_visualizations: bool = False,
    ) -> Dict[str, Any]:
        """运行完整评估。
        instances: 若传入则直接使用（用于消融等多配置共用同一子集）；否则从 dataset_type 加载。
        sample_seed: 与 sample_size 联用，从全量中无放回随机抽样子集；为 None 时取前 sample_size 条。
        run_tag: 写入 evaluation_config，便于区分多次运行。
        skip_visualizations: 为 True 时跳过绘图（批量消融时加速）。
        """
        logger.info(f"开始LongMemEval评估（New Memory），数据集类型: {dataset_type}")
        logger.info(f"使用模型: {self.config['evaluation_llm']}")
        self.detailed_results = []
        if instances is not None:
            instances = list(instances)
            logger.info("使用外部传入的 %d 条实例", len(instances))
        else:
            instances = self.load_benchmark_data(dataset_type)
            if sample_size and sample_size < len(instances):
                if sample_seed is not None:
                    rng = np.random.default_rng(sample_seed)
                    pick = rng.choice(len(instances), size=sample_size, replace=False)
                    instances = [instances[i] for i in pick]
                    logger.info("按种子 %d 随机采样 %d 个实例", sample_seed, sample_size)
                else:
                    instances = instances[:sample_size]
                    logger.info(f"采样 {sample_size} 个实例进行评估")
        results_by_type = defaultdict(list)
        overall_results = []
        pbar = tqdm(instances, desc="评估进度", unit="实例")
        for i, instance in enumerate(instances):
            pbar.set_description(f"评估进度 (实例 {i+1}/{len(instances)})")
            pbar.refresh()
            result = self.evaluate_single_instance(instance)
            results_by_type[instance["question_type"]].append(result)
            overall_results.append(result)
            pbar.update(1)
        pbar.close()
        final_metrics = self._calculate_overall_metrics(
            overall_results, results_by_type, run_tag=run_tag, dataset_type=dataset_type
        )
        self._save_results(final_metrics, dataset_type, len(instances))
        if not skip_visualizations:
            self._generate_visualizations(final_metrics, dataset_type)
        logger.info("评估完成")
        return final_metrics
    
    def _calculate_overall_metrics(
        self,
        overall_results: List[Dict],
        results_by_type: Dict,
        run_tag: Optional[str] = None,
        dataset_type: str = "oracle",
    ) -> Dict[str, Any]:
        metrics = {
            "overall": self._calculate_metrics_for_group(overall_results),
            "by_question_type": {},
            "evaluation_config": {
                "model": self.config["evaluation_llm"],
                "dataset_type": dataset_type,
                "memory_system": "memory_manager",
                "timestamp": datetime.now().isoformat(),
            },
            "total_instances": len(overall_results)
        }
        if run_tag:
            metrics["evaluation_config"]["run_tag"] = run_tag
        for q_type, results in results_by_type.items():
            metrics["by_question_type"][q_type] = self._calculate_metrics_for_group(results)
        return metrics
    
    def _calculate_metrics_for_group(self, results: List[Dict]) -> Dict[str, float]:
        if not results:
            return {}
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            return {}
        avg_retrieval_recall = np.mean([r.get("retrieval_metrics", {}).get("recall@k", 0) for r in valid_results])
        avg_qa_correctness = np.mean([r.get("qa_metrics", {}).get("correctness", 0) for r in valid_results])
        correctness_scores = [r.get("qa_metrics", {}).get("correctness", 0) for r in valid_results]
        metrics = {
            "avg_retrieval_recall": float(avg_retrieval_recall),
            "avg_qa_correctness": float(avg_qa_correctness),
            "success_rate": len(valid_results) / len(results),
            "total_valid_instances": len(valid_results)
        }
        if correctness_scores:
            arr = np.array(correctness_scores, dtype=float)
            metrics.update({
                "correctness_std": float(np.std(arr)),
                "correctness_min": float(np.min(arr)),
                "correctness_max": float(np.max(arr)),
                "correctness_median": float(np.median(arr))
            })
        return metrics
    
    def _save_results(self, metrics: Dict[str, Any], dataset_type: str, total_instances: int) -> None:
        os.makedirs(self.config["output_dir"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_filename = f"longmemeval_new_summary_{dataset_type}_{timestamp}.json"
        metrics_filepath = os.path.join(self.config["output_dir"], metrics_filename)
        with open(metrics_filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"指标汇总已保存至: {metrics_filepath}")
        should_save_detailed = (
            self.config.get("save_detailed_results", True) and 
            total_instances <= self.config.get("detailed_results_threshold", 50)
        )
        if should_save_detailed and self.detailed_results:
            detailed_filename = f"longmemeval_new_detailed_{dataset_type}_{timestamp}.json"
            detailed_filepath = os.path.join(self.config["output_dir"], detailed_filename)
            detailed_output = {
                "metadata": {
                    "dataset_type": dataset_type,
                    "total_instances": total_instances,
                    "evaluation_model": self.config["evaluation_llm"],
                    "memory_system": "memory_manager",
                    "timestamp": datetime.now().isoformat(),
                },
                "detailed_results": self.detailed_results
            }
            with open(detailed_filepath, 'w', encoding='utf-8') as f:
                json.dump(detailed_output, f, ensure_ascii=False, indent=2)
            logger.info(f"详细评估过程已保存至: {detailed_filepath}")
    
    def _generate_visualizations(self, metrics: Dict[str, Any], dataset_type: str) -> None:
        try:
            self._create_metrics_plot(metrics, dataset_type)
            self._create_question_type_comparison(metrics, dataset_type)
        except Exception as e:
            logger.warning(f"生成可视化失败: {e}")
    
    def _create_metrics_plot(self, metrics: Dict[str, Any], dataset_type: str) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        retrieval_data = []
        labels = []
        for q_type, type_metrics in metrics["by_question_type"].items():
            retrieval_data.append(type_metrics["avg_retrieval_recall"])
            labels.append(q_type)
        ax1.bar(labels, retrieval_data, color='skyblue')
        ax1.set_title('Average Recall@k for All Question Types')
        ax1.set_ylabel('Recall@k')
        ax1.tick_params(axis='x', rotation=45)
        qa_data = [metrics["by_question_type"][q]["avg_qa_correctness"] for q in metrics["by_question_type"]]
        ax2.bar(labels, qa_data, color='lightcoral')
        ax2.set_title('Average QA Correctness for All Question Types')
        ax2.set_ylabel('Correctness')
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.config["output_dir"], f"metrics_comparison_new_{dataset_type}_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"指标对比图已保存至: {plot_path}")
    
    def _create_question_type_comparison(self, metrics: Dict[str, Any], dataset_type: str) -> None:
        question_types = list(metrics["by_question_type"].keys())
        metrics_list = ["avg_retrieval_recall", "avg_qa_correctness", "success_rate"]
        data = np.zeros((len(metrics_list), len(question_types)))
        for i, metric in enumerate(metrics_list):
            for j, q_type in enumerate(question_types):
                data[i, j] = metrics["by_question_type"][q_type][metric]
        plt.figure(figsize=(10, 6))
        sns.heatmap(data, annot=True, fmt='.3f', xticklabels=question_types, 
                   yticklabels=metrics_list, cmap='YlOrRd')
        plt.title('Heatmap of Question Type Metrics (New Memory)')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        heatmap_path = os.path.join(self.config["output_dir"], f"heatmap_comparison_new_{dataset_type}_{timestamp}.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='LongMemEval 记忆系统评估（New Memory / memory_manager）')
    parser.add_argument('--dataset', type=str, default='oracle', choices=['oracle', 's'], help='数据集类型')
    parser.add_argument('--sample_size', type=int, default=500, help='采样数量（用于快速测试）')
    parser.add_argument('--config', type=str, default='eval_config.json', help='配置文件路径')
    parser.add_argument('--model', type=str, default='gpt-oss:20b', help='Ollama模型名称')
    parser.add_argument('--save_detailed', action='store_true', help='强制保存详细评估结果')
    parser.add_argument('--output_dir', type=str, default='results/0314', help='结果输出目录')
    parser.add_argument('--tune_threshold', action='store_true', help='先用训练集异步调参（强化学习式）再评估')
    parser.add_argument('--tune_workers', type=int, default=4, help='阈值调参并行 worker 数')
    parser.add_argument('--tune_train_ratio', type=float, default=None, help='调参时训练集比例 (0~1)，不传则用 config 的 tune_train_ratio')
    parser.add_argument('--tune_seed', type=int, default=None, help='调参划分的随机种子，不传则用 config 的 tune_seed')
    
    args = parser.parse_args()
    
    from memory_manager import VectorMemoryManager as MemoryManager
    evaluator = LongMemEvalEvaluator(MemoryManager(), args.config)
    evaluator.config["output_dir"] = args.output_dir
    if args.model:
        evaluator.config["evaluation_llm"] = args.model
    if args.tune_threshold:
        evaluator.run_threshold_tuning(
            dataset_type=args.dataset,
            train_ratio=args.tune_train_ratio,
            tune_seed=args.tune_seed,
            max_workers=args.tune_workers,
        )
    retrieval_params = {
        k: evaluator.config[k]
        for k in ("min_summary_similarity", "knowledge_min_score")
        if k in evaluator.config
    }
    memory_manager = MemoryManager(retrieval_params=retrieval_params if retrieval_params else None)
    evaluator.memory_manager = memory_manager
    evaluator.config["output_dir"] = args.output_dir
    
    if args.model:
        evaluator.config["evaluation_llm"] = args.model
    if args.save_detailed:
        evaluator.config["save_detailed_results"] = True
        evaluator.config["detailed_results_threshold"] = 10000
    
    results = evaluator.run_evaluation(
        dataset_type=args.dataset,
        sample_size=args.sample_size
    )
    
    print("\n" + "="*60)
    print("LongMemEval 评估结果摘要（New Memory / memory_manager）")
    print("="*60)
    overall = results["overall"]
    config = results["evaluation_config"]
    print(f"评估模型: {config['model']}")
    print(f"记忆系统: {config.get('memory_system', 'memory_manager')}")
    print(f"数据集类型: {config['dataset_type']}")
    print(f"评估时间: {config['timestamp']}")
    print(f"总实例数: {results['total_instances']}")
    print(f"总体检索召回率: {overall['avg_retrieval_recall']:.3f}")
    print(f"总体QA正确率: {overall['avg_qa_correctness']:.3f}")
    print(f"成功率: {overall['success_rate']:.3f}")
    print("\n各问题类型表现:")
    print("-" * 50)
    for q_type, metrics in results["by_question_type"].items():
        print(f"{q_type:25} | 召回率: {metrics['avg_retrieval_recall']:.3f} | "
              f"正确率: {metrics['avg_qa_correctness']:.3f} | "
              f"成功率: {metrics['success_rate']:.3f}")
    print("\n详细结果已保存至:", evaluator.config["output_dir"])

if __name__ == "__main__":
    main()
