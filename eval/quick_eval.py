import os
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datetime import datetime
from memory_manager import VectorMemoryManager
from eval_new import OllamaClient  # 使用您提供的Ollama客户端实现

RESULTS_DIR = "results/0313"
MODEL_NAME = "gpt-oss:20b"  # 默认使用您的配置模型
TEMPERATURE = 0.3           # 默认温度参数
MAX_ANSWER_LENGTH = 200     # 答案最大长度

class QuickEvaluator:
    def __init__(self):
        self.memory_manager = VectorMemoryManager()
        self.ollama_client = OllamaClient()  # 使用您提供的Ollama客户端
        self._setup_results_dir()
    
    def _setup_results_dir(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"结果将保存至: {RESULTS_DIR}")
    
    def _generate_answer(self, query, retrieved_results):
        """使用本地Ollama生成答案"""
        context = "\n\n".join([
            f"{item.get('type', '未知类型')} - {item['tid']}\n{item['text']}"
            for item in retrieved_results[:5]  # 限制上下文长度
        ])
        
        prompt = f"""
        请根据提供的记忆内容回答用户的问题：

        【问题】
        {query}

        【记忆上下文】
        {context}

        【回答要求】
        1. 仅使用提供的记忆内容作答
        2. 保持口语化，避免使用Markdown格式
        3. 答案控制在{MAX_ANSWER_LENGTH}字以内
        """.strip()
        
        try:
            # 直接调用您提供的OllamaClient生成答案
            response = self.ollama_client.generate_response(
                model=MODEL_NAME,
                prompt=prompt,
                temperature=TEMPERATURE
            )
            return response.strip()
        except Exception as e:
            print(f"答案生成失败: {str(e)}", file=sys.stderr)
            return "很抱歉，目前无法生成答案。"

    def run(self):
        print("记忆系统已启动，当前已加载对话历史。")
        print(f"使用模型: {MODEL_NAME}")
        print(f"温度参数: {TEMPERATURE}")
        print(f"结果保存目录: {RESULTS_DIR}\n")
        
        while True:
            query = input("\n请输入查询内容（输入exit退出）：").strip()
            if query.lower() == 'exit':
                break
            
            try:
                top_k = int(input("请输入需要返回的结果数量（Top-K）："))
                if top_k < 1 or top_k > 100:
                    print("请输入1-100之间的整数")
                    continue
            except ValueError:
                print("请输入有效的数字")
                continue
            
            try:
                results = self.memory_manager.search(
                    query,
                    top_k=top_k,
                    is_temporal_task=False  # 关闭时序推理模式
                )
            except Exception as e:
                print(f"检索失败: {str(e)}", file=sys.stderr)
                continue
            
            if not results:
                print("未找到相关结果")
                continue
            
            # 生成答案
            answer = self._generate_answer(query, results)
            print(f"\n=== 生成的答案 ===")
            print(answer)
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{RESULTS_DIR}/query_results_{timestamp}.json"
            
            save_data = {
                "query": query,
                "top_k": top_k,
                "results": results,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                print(f"完整结果已保存至: {filename}")
            except Exception as e:
                print(f"保存结果失败: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    evaluator = QuickEvaluator()
    evaluator.run()