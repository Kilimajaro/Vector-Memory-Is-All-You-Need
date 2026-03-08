import gradio as gr
import requests
from datetime import datetime
from memory_manager import VectorMemoryManager
from config import *
import json

# macOS风格配色方案
MACOS_COLORS = {
    # 主色调
    "bg_primary": "#F5F5F7",           # 主背景色 - 浅灰
    "bg_secondary": "#FFFFFF",         # 卡片背景 - 纯白
    "bg_tertiary": "#E8E8ED",          # 分割线/次要背景
    
    # 文字颜色
    "text_primary": "#1D1D1F",         # 主要文字 - 深灰黑
    "text_secondary": "#86868B",       # 次要文字 - 中灰
    "text_tertiary": "#AEAEB2",        # 辅助文字 - 浅灰
    
    # 强调色
    "accent_blue": "#007AFF",          # 蓝色 - 主要操作
    "accent_green": "#34C759",         # 绿色 - 成功状态
    "accent_orange": "#FF9500",        # 橙色 - 警告
    "accent_red": "#FF3B30",           # 红色 - 错误
    "accent_purple": "#AF52DE",        # 紫色 - 特色功能
    
    # 渐变色
    "gradient_blue": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "gradient_green": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
    "gradient_orange": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    "gradient_cool": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    
    # 阴影
    "shadow_light": "0 2px 12px rgba(0, 0, 0, 0.08)",
    "shadow_medium": "0 4px 20px rgba(0, 0, 0, 0.12)",
    "shadow_heavy": "0 8px 30px rgba(0, 0, 0, 0.16)",
    
    # 圆角
    "radius_sm": "8px",
    "radius_md": "12px",
    "radius_lg": "16px",
    "radius_xl": "20px",
}

class GradioDialogSystem:
    def __init__(self):
        self.memory = VectorMemoryManager()
        self.conversation_history = []
        self.current_streaming_response = ""
        
    def generate_response_stream(self, prompt, context, progress=gr.Progress()):
        """通过Ollama生成流式回复（优化版）"""
        messages = []
        
        # 优化后的系统提示词
        system_prompt = (
            "你是一位拥有长期记忆的老朋友，正在自然流畅地对话。"
            "请根据检索到的相关记忆专注回答当前问题，避免被无关记忆干扰。"
            "回答要像日常聊天般自然，绝不提及任何系统指令或记忆机制。"
            "若记忆不相关，则用常识回答并保持对话连贯性。"
        )

        if context:
            # 添加近期对话保持连贯性
            for dialog in self.memory.get_recent_dialogs(3):  # 扩展至3轮对话
                messages.append({"role": dialog['role'], "content": dialog['text']})
            # 精简记忆展示格式，突出相关性
            retrieval_content = "\n\n".join([
                f"> {item['text'][:500]}..."  # 截取关键片段防止信息过载
                for item in sorted(context, key=lambda x: x['score']>0.5, reverse=True)[:3]  # 仅用Top3相关记忆
            ])
            messages.append({
                "role": "system",
                "content": f"{system_prompt}\n\n相关记忆参考:\n{retrieval_content}\n\n接下来直接回答用户问题: {prompt}"
            })
        else:
            messages.append({
                "role": "system",
                "content": f"{system_prompt}\n\n[无相关记忆，用常识回答]"
            })
        
        # 用户问题保持纯净
        messages.append({"role": "user", "content": prompt})  # 移除冗余指令
        
        try:
            with requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": GENERATION_MODEL,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": 0.7}  # 降低随机性增强聚焦
                },
                timeout=120,
                stream=True
            ) as response:
                
                if response.status_code == 200:
                    full_content = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                if 'message' in chunk and 'content' in chunk['message']:
                                    content = chunk['message']['content']
                                    full_content += content
                                    yield content, full_content
                            except json.JSONDecodeError:
                                continue
                    yield None, full_content
                else:
                    yield f"⚠️ 服务暂不可用({response.status_code})", f"⚠️ 服务暂不可用({response.status_code})"
                    
        except Exception as e:
            yield f"⚠️ 思考中断: {str(e)[:50]}", f"⚠️ 思考中断: {str(e)[:50]}"
    
    def process_query(self, user_input, progress=gr.Progress()):
        progress(0.1, desc="开始处理查询...")
        
        progress(0.3, desc="进行记忆检索...")
        search_results = self.memory.search(user_input)
        
        search_html = self.build_pyramid_memory_display(search_results)
        
        progress(0.6, desc="生成回复中...")
        
        self.current_streaming_response = ""
        
        progress(0.8, desc="准备保存...")
        
        self.conversation_history.append({"role": "user", "content": user_input})
        
        progress(1.0, desc="完成!")
        return user_input, search_html, self._get_conversation_history_html(), True
    
    def build_pyramid_memory_display(self, search_results):
        """构建真正的金字塔式记忆层次可视化展示 - macOS风格"""
        if not search_results:
            return f"""
            <div style="
                text-align: center; 
                padding: 60px 40px; 
                color: {MACOS_COLORS['text_tertiary']};
                background: {MACOS_COLORS['bg_secondary']};
                border-radius: {MACOS_COLORS['radius_lg']};
                border: 1px dashed {MACOS_COLORS['bg_tertiary']};
            ">
                <div style="font-size: 48px; margin-bottom: 16px;">📚</div>
                <p style="font-size: 1.1em; margin-bottom: 8px; color: {MACOS_COLORS['text_secondary']};">未检索到相关记忆</p>
                <p style="font-size: 0.9em; color: {MACOS_COLORS['text_tertiary']};">尝试换一个关键词或问题</p>
            </div>
            """
        
        knowledge_items = [r for r in search_results if r['type'] == 'knowledge_item']
        paragraphs = [r for r in search_results if r['type'] == 'paragraph']
        sentences = [r for r in search_results if r['type'] == 'sentence']
        
        total_items = len(knowledge_items) + len(paragraphs) + len(sentences)
        
        if knowledge_items:
            knowledge_width = max(30, min(80, 100 - (len(paragraphs) + len(sentences)) * 5))
        else:
            knowledge_width = 0
            
        if paragraphs:
            paragraph_width = max(50, min(90, 100 - len(sentences) * 3))
        else:
            paragraph_width = 0
            
        sentence_width = 100
        
        # 构建金字塔HTML
        pyramid_parts = []
        
        # CSS样式 - macOS风格
        css_style = f"""
        <style>
            @keyframes slideUp {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.02); }}
            }}
            
            .pyramid-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                margin: 24px 0;
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
                animation: fadeIn 0.5s ease-out;
            }}
            
            .pyramid-title {{
                text-align: center;
                margin-bottom: 28px;
                animation: slideUp 0.6s ease-out;
            }}
            
            .pyramid-title h4 {{
                font-size: 1.35em;
                font-weight: 600;
                color: {MACOS_COLORS['text_primary']};
                margin-bottom: 6px;
                letter-spacing: -0.02em;
            }}
            
            .pyramid-title p {{
                color: {MACOS_COLORS['text_secondary']};
                font-size: 0.92em;
            }}
            
            .pyramid-level {{
                width: VAR_WIDTH%;
                margin: 0 auto;
                border-radius: {MACOS_COLORS['radius_md']};
                overflow: hidden;
                box-shadow: {MACOS_COLORS['shadow_light']};
                transition: all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94);
                position: relative;
                animation: slideUp 0.5s ease-out both;
                background: {MACOS_COLORS['bg_secondary']};
            }}
            
            .pyramid-level:nth-child(2) {{ animation-delay: 0.1s; }}
            .pyramid-level:nth-child(4) {{ animation-delay: 0.15s; }}
            .pyramid-level:nth-child(6) {{ animation-delay: 0.2s; }}
            
            .pyramid-level:hover {{
                transform: translateY(-6px);
                box-shadow: {MACOS_COLORS['shadow_medium']};
            }}
            
            .level-header {{
                padding: 14px 22px;
                color: white;
                font-weight: 500;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: filter 0.2s ease;
                font-size: 0.95em;
            }}
            
            .level-header:hover {{
                filter: brightness(1.08);
            }}
            
            .level-header span:first-child {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .level-badge {{
                background: rgba(255, 255, 255, 0.22);
                padding: 4px 10px;
                border-radius: 14px;
                font-size: 0.82em;
                font-weight: 500;
            }}
            
            .toggle-icon::after {{
                content: '›';
                font-size: 1.2em;
                transition: transform 0.3s ease;
                display: inline-block;
            }}
            
            .collapsed .toggle-icon::after {{
                transform: rotate(90deg);
            }}
            
            .collapsed .level-content {{
                display: none;
            }}
            
            .level-content {{
                padding: 18px;
                background: {MACOS_COLORS['bg_secondary']};
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
                gap: 14px;
            }}
            
            .memory-card {{
                border: 1px solid {MACOS_COLORS['bg_tertiary']};
                border-radius: {MACOS_COLORS['radius_sm']};
                padding: 14px;
                background: {MACOS_COLORS['bg_secondary']};
                transition: all 0.25s ease;
                position: relative;
                overflow: hidden;
                animation: fadeIn 0.4s ease-out both;
            }}
            
            .memory-card:nth-child(odd) {{ animation-delay: 0.05s; }}
            .memory-card:nth-child(even) {{ animation-delay: 0.1s; }}
            
            .memory-card:hover {{
                background-color: #FAFBFC;
                transform: translateY(-2px);
                box-shadow: {MACOS_COLORS['shadow_light']};
            }}
            
            .memory-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 3px;
                height: 100%;
                background: var(--accent-color, {MACOS_COLORS['accent_blue']});
                border-radius: {MACOS_COLORS['radius_sm']} 0 0 {MACOS_COLORS['radius_sm']};
            }}
            
            .memory-type {{
                font-size: 0.78em;
                color: {MACOS_COLORS['text_tertiary']};
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                font-weight: 500;
            }}
            
            .memory-score {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.72em;
                font-weight: 600;
                margin-bottom: 10px;
            }}
            
            .high-score {{ 
                background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); 
                color: #065F46; 
            }}
            
            .medium-score {{ 
                background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); 
                color: #92400E; 
            }}
            
            .low-score {{ 
                background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%); 
                color: #991B1B; 
            }}
            
            .memory-text {{
                margin-top: 10px;
                font-size: 0.88em;
                line-height: 1.55;
                color: {MACOS_COLORS['text_secondary']};
            }}
            
            .pyramid-connector {{
                width: 2px;
                height: 12px;
                background: linear-gradient(180deg, transparent, {MACOS_COLORS['bg_tertiary']});
                margin: 0 auto;
            }}
            
            /* 层级特定颜色 */
            .level-knowledge .level-header {{
                background: linear-gradient(135deg, {MACOS_COLORS['accent_blue']} 0%, #5856D6 100%);
            }}
            .level-knowledge {{ --accent-color: {MACOS_COLORS['accent_blue']}; }}
            
            .level-paragraph .level-header {{
                background: linear-gradient(135deg, {MACOS_COLORS['accent_green']} 0%, #30B94D 100%);
            }}
            .level-paragraph {{ --accent-color: {MACOS_COLORS['accent_green']}; }}
            
            .level-sentence .level-header {{
                background: linear-gradient(135deg, {MACOS_COLORS['accent_orange']} 0%, #FF7043 100%);
            }}
            .level-sentence {{ --accent-color: {MACOS_COLORS['accent_orange']}; }}
        </style>
        """
        pyramid_parts.append(css_style)
        
        # 标题部分
        title_html = f"""
        <div class="pyramid-container">
            <div class="pyramid-title">
                <h4>🏛️ 记忆检索结果</h4>
                <p>共检索到 {total_items} 条相关记忆 · 金字塔层次结构</p>
            </div>
        """
        pyramid_parts.append(title_html)
        
        # 知识级
        if knowledge_items:
            knowledge_width_value = knowledge_width if knowledge_width > 0 else 40
            level_html = f"""
            <div class="pyramid-level level-knowledge" style="width: {knowledge_width_value}%;">
                <div class="level-header" onclick="this.parentElement.classList.toggle('collapsed')">
                    <span><span class="level-badge">🎯 顶层</span> 知识级 · {len(knowledge_items)}条</span>
                    <span class="toggle-icon"></span>
                </div>
                <div class="level-content">
            """
            for i, item in enumerate(sorted(knowledge_items, key=lambda x: x['score'], reverse=True), 1):
                score_class = self._get_score_class(item['score'])
                text_preview = item['text'][:250] + ('...' if len(item['text']) > 250 else '')
                level_html += f"""
                    <div class="memory-card">
                        <div class="memory-type">📖 知识项 #{i}</div>
                        <div>
                            <span class="memory-score {score_class}">相似度: {item['score']:.3f}</span>
                        </div>
                        <div class="memory-text">{text_preview}</div>
                    </div>
                """
            level_html += "</div></div>"
            pyramid_parts.append(level_html)
            
            if paragraphs or sentences:
                pyramid_parts.append('<div class="pyramid-connector"></div>\n')
        
        # 段落级
        if paragraphs:
            paragraph_width_value = paragraph_width if paragraph_width > 0 else 60
            level_html = f"""
            <div class="pyramid-level level-paragraph" style="width: {paragraph_width_value}%;">
                <div class="level-header" onclick="this.parentElement.classList.toggle('collapsed')">
                    <span><span class="level-badge">📄 中层</span> 段落级 · {len(paragraphs)}条</span>
                    <span class="toggle-icon"></span>
                </div>
                <div class="level-content">
            """
            for i, item in enumerate(sorted(paragraphs, key=lambda x: x['score'], reverse=True), 1):
                score_class = self._get_score_class(item['score'])
                text_preview = item['text'][:250] + ('...' if len(item['text']) > 250 else '')
                level_html += f"""
                    <div class="memory-card">
                        <div class="memory-type">📝 段落 #{i}</div>
                        <div>
                            <span class="memory-score {score_class}">相似度: {item['score']:.3f}</span>
                        </div>
                        <div class="memory-text">{text_preview}</div>
                    </div>
                """
            level_html += "</div></div>"
            pyramid_parts.append(level_html)
            
            if sentences:
                pyramid_parts.append('<div class="pyramid-connector"></div>\n')
        
        # 句子级
        if sentences:
            level_html = f"""
            <div class="pyramid-level level-sentence" style="width: 100%;">
                <div class="level-header" onclick="this.parentElement.classList.toggle('collapsed')">
                    <span><span class="level-badge">💬 底层</span> 句子级 · {len(sentences)}条</span>
                    <span class="toggle-icon"></span>
                </div>
                <div class="level-content">
            """
            for i, item in enumerate(sorted(sentences, key=lambda x: x['score'], reverse=True), 1):
                score_class = self._get_score_class(item['score'])
                text_preview = item['text'][:250] + ('...' if len(item['text']) > 250 else '')
                level_html += f"""
                    <div class="memory-card">
                        <div class="memory-type">💬 句子 #{i}</div>
                        <div>
                            <span class="memory-score {score_class}">相似度: {item['score']:.3f}</span>
                        </div>
                        <div class="memory-text">{text_preview}</div>
                    </div>
                """
            level_html += "</div></div>"
            pyramid_parts.append(level_html)
        
        pyramid_parts.append("</div>")
        
        return ''.join(pyramid_parts)
    
    def get_system_status_html(self):
        try:
            dialogs_count = len(self.memory.get_recent_dialogs(1000))
            vector_count = self.memory.vector_index.ntotal if hasattr(self.memory, 'vector_index') else 0
            
            status_html = f"""
            <div style="
                background: {MACOS_COLORS['bg_secondary']};
                border-radius: {MACOS_COLORS['radius_lg']};
                padding: 22px;
                box-shadow: {MACOS_COLORS['shadow_light']};
                border: 1px solid {MACOS_COLORS['bg_tertiary']};
            ">
                <h4 style="
                    margin-top: 0; 
                    margin-bottom: 18px; 
                    font-size: 1.15em; 
                    font-weight: 600;
                    color: {MACOS_COLORS['text_primary']};
                    display: flex;
                    align-items: center;
                    gap: 8px;
                ">
                    📊 系统状态
                </h4>
                <div style="display: grid; gap: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid {MACOS_COLORS['bg_tertiary']};">
                        <span style="color: {MACOS_COLORS['text_secondary']}; font-size: 0.9em;">🤖 当前模型</span>
                        <span style="color: {MACOS_COLORS['text_primary']}; font-weight: 500; font-size: 0.9em;">{GENERATION_MODEL}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid {MACOS_COLORS['bg_tertiary']};">
                        <span style="color: {MACOS_COLORS['text_secondary']}; font-size: 0.9em;">🔢 记忆向量数</span>
                        <span style="color: {MACOS_COLORS['accent_blue']}; font-weight: 600; font-size: 0.9em;">{vector_count:,}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid {MACOS_COLORS['bg_tertiary']};">
                        <span style="color: {MACOS_COLORS['text_secondary']}; font-size: 0.9em;">💬 对话记录数</span>
                        <span style="color: {MACOS_COLORS['accent_green']}; font-weight: 600; font-size: 0.9em;">{dialogs_count}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid {MACOS_COLORS['bg_tertiary']};">
                        <span style="color: {MACOS_COLORS['text_secondary']}; font-size: 0.9em;">🧮 嵌入模型</span>
                        <span style="color: {MACOS_COLORS['text_primary']}; font-weight: 500; font-size: 0.9em;">{EMBEDDING_MODEL}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0;">
                        <span style="color: {MACOS_COLORS['text_secondary']}; font-size: 0.9em;">⏰ 最后更新</span>
                        <span style="color: {MACOS_COLORS['text_primary']}; font-weight: 500; font-size: 0.9em;">{datetime.now().strftime('%m/%d %H:%M')}</span>
                    </div>
                </div>
            </div>
            """
            return status_html
        except Exception as e:
            return f"<p style='color: {MACOS_COLORS['accent_red']}; text-align: center;'>❌ 获取状态失败</p>"
    
    def clear_conversation(self):
        self.conversation_history = []
        return "✅ 对话历史已清空", self._get_conversation_history_html()

    def _get_conversation_history_html(self):
        if not self.conversation_history:
            return f"""
            <div style="
                text-align: center; 
                padding: 50px 30px; 
                color: {MACOS_COLORS['text_tertiary']};
                background: {MACOS_COLORS['bg_secondary']};
                border-radius: {MACOS_COLORS['radius_lg']};
                border: 1px dashed {MACOS_COLORS['bg_tertiary']};
            ">
                <div style="font-size: 42px; margin-bottom: 14px; opacity: 0.6;">🗨️</div>
                <p style="font-size: 1.05em; color: {MACOS_COLORS['text_secondary']};">暂无对话历史</p>
                <p style="font-size: 0.88em; margin-top: 6px;">开始对话后，这里会显示您的对话记录</p>
            </div>
            """
        
        history_html = f"""
        <div style="
            max-height: 420px; 
            overflow-y: auto; 
            padding: 16px; 
            border-radius: {MACOS_COLORS['radius_lg']};
            background: {MACOS_COLORS['bg_secondary']};
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.04);
            border: 1px solid {MACOS_COLORS['bg_tertiary']};
        ">
        """
        
        for i, dialog in enumerate(self.conversation_history):
            role = dialog['role']
            content = dialog['content']
            if role == "user":
                bg_color = MACOS_COLORS['gradient_blue']
                role_icon = "👤"
                role_name = "您"
            else:
                bg_color = MACOS_COLORS['gradient_green']
                role_icon = "🤖"
                role_name = "AI助手"
            
            safe_content = content.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            
            history_html += f"""
            <div style="
                margin-bottom: 16px; 
                padding: 14px 16px; 
                border-radius: {MACOS_COLORS['radius_md']}; 
                background: {bg_color}; 
                color: white; 
                box-shadow: {MACOS_COLORS['shadow_light']};
                animation: slideUp 0.3s ease-out;
            ">
                <div style="display: flex; align-items: center; margin-bottom: 8px; gap: 8px;">
                    <span style="font-size: 1.1em; opacity: 0.95;">{role_icon}</span>
                    <strong style="font-size: 0.9em; font-weight: 500; opacity: 0.95;">{role_name}</strong>
                </div>
                <div style="font-size: 0.93em; line-height: 1.6; opacity: 0.92; word-wrap: break-word;">{safe_content}</div>
            </div>
            """
        
        history_html += """
        <style>
            @keyframes slideUp {
                from { opacity: 0; transform: translateY(12px); }
                to { opacity: 1; transform: translateY(0); }
            }
            div::-webkit-scrollbar {
                width: 6px;
            }
            div::-webkit-scrollbar-track {
                background: transparent;
            }
            div::-webkit-scrollbar-thumb {
                background: {MACOS_COLORS['bg_tertiary']};
                border-radius: 3px;
            }
            div::-webkit-scrollbar-thumb:hover {
                background: {MACOS_COLORS['text_tertiary']};
            }
        </style>
        </div>
        """
        return history_html


def create_interface():
    system = GradioDialogSystem()
    
    # macOS风格主题
    macos_theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="green",
        neutral_hue="gray",
        font=["-apple-system", "BlinkMacSystemFont", "SF Pro Display", "Segoe UI", "Roboto", "sans-serif"],
    ).set(
        body_background_fill="#F5F5F7",
        block_background_fill="#FFFFFF",
        block_border_width="1px",
        block_border_color="#E8E8ED",
        block_radius="12px",
        block_shadow="0 2px 12px rgba(0, 0, 0, 0.08)",
        input_background_fill="#FFFFFF",
        input_border_color="#D2D2D7",
        input_radius="10px",
        button_primary_background_fill="#007AFF",
        button_primary_background_fill_hover="#0062CC",
        button_primary_text_color="white",
        button_secondary_background_fill="#F5F5F7",
        button_secondary_background_fill_hover="#E8E8ED",
        button_secondary_text_color="#1D1D1F",
        button_radius="10px",
        color_accent_soft="#007AFF20",
    )
    
    with gr.Blocks(theme=macos_theme, title="智能记忆对话系统") as demo:
        gr.Markdown(f"""
        <div style="
            text-align: center; 
            padding: 30px 20px 40px;
            background: linear-gradient(180deg, #FFFFFF 0%, #F5F5F7 100%);
            border-radius: 0 0 24px 24px;
            margin: -20px -20px 30px -20px;
        ">
            <h1 style="
                font-size: 2.2em; 
                font-weight: 700; 
                color: {MACOS_COLORS['text_primary']};
                margin-bottom: 10px;
                letter-spacing: -0.02em;
            ">🧠 智能记忆对话系统</h1>
            <p style="
                font-size: 1.05em; 
                color: {MACOS_COLORS['text_secondary']};
                font-weight: 400;
            ">基于向量中心的记忆模块 · 实现长期对话记忆管理</p>
        </div>
        """)
        
        streaming_state = gr.State(False)
        current_response = gr.State("")
        
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                # 对话显示区域
                conversation_display = gr.HTML(
                    value=system._get_conversation_history_html(),
                    label="对话历史",
                    elem_id="conversation-display"
                )
                
                # 用户输入区域
                with gr.Group(elem_classes="input-group"):
                    with gr.Row():
                        user_input = gr.Textbox(
                            lines=2,
                            placeholder="输入您的问题，按 Enter 或点击发送...",
                            label="",
                            scale=4,
                            container=False,
                        )
                        submit_btn = gr.Button(
                            "发送", 
                            variant="primary", 
                            scale=1,
                            elem_classes="send-btn"
                        )
                
                # 控制按钮
                with gr.Row(elem_classes="control-buttons"):
                    clear_btn = gr.Button(
                        "🗑️ 清空对话", 
                        variant="secondary",
                        elem_classes="control-btn"
                    )
                    status_btn = gr.Button(
                        "🔄 刷新状态", 
                        variant="secondary",
                        elem_classes="control-btn"
                    )
            
            with gr.Column(scale=1):
                # 系统状态显示
                status_display = gr.HTML(
                    value=system.get_system_status_html(),
                    label="系统状态",
                    elem_id="status-display"
                )
                
                # 记忆检索结果显示
                search_display = gr.HTML(
                    label="记忆检索结果",
                    value=f"""
                    <div style="
                        text-align: center; 
                        padding: 40px 20px; 
                        color: {MACOS_COLORS['text_tertiary']};
                        background: {MACOS_COLORS['bg_secondary']};
                        border-radius: {MACOS_COLORS['radius_lg']};
                        border: 1px dashed {MACOS_COLORS['bg_tertiary']};
                    ">
                        <div style="font-size: 44px; margin-bottom: 14px; opacity: 0.5;">🔍</div>
                        <p style="font-size: 1em; color: {MACOS_COLORS['text_secondary']};">等待查询...</p>
                    </div>
                    """,
                    elem_id="search-display"
                )
        
# 自定义CSS注入
        gr.HTML(f"""
        <style>
            /* macOS风格全局样式 */
            .gradio-container {{
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif !important;
            }}
            
            /* 输入框聚焦效果 */
            .input-group textarea {{
                border-radius: 12px !important;
                border: 2px solid #E8E8ED !important;
                transition: all 0.2s ease !important;
                padding: 14px 16px !important;
                font-size: 15px !important;
            }}
            
            .input-group textarea:focus {{
                border-color: {MACOS_COLORS['accent_blue']} !important;
                box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.1) !important;
            }}
            
            /* 按钮悬停动画 */
            .send-btn {{
                transition: all 0.2s ease !important;
                font-weight: 500 !important;
            }}
            
            .send-btn:hover {{
                transform: scale(1.02) !important;
            }}
            
            .send-btn:active {{
                transform: scale(0.98) !important;
            }}
            
            .control-btn {{
                transition: all 0.2s ease !important;
                font-size: 13px !important;
            }}
            
            .control-btn:hover {{
                transform: translateY(-1px) !important;
            }}
            
            /* 标签样式 */
            .label {{
                font-size: 12px !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.05em !important;
                color: {MACOS_COLORS['text_secondary']} !important;
                margin-bottom: 8px !important;
            }}
            
            /* 卡片容器 */
            .panel {{
                background: {MACOS_COLORS['bg_secondary']} !important;
                border-radius: 16px !important;
                box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06) !important;
                border: 1px solid {MACOS_COLORS['bg_tertiary']} !important;
            }}
            
            /* 滚动条美化 */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: transparent;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: {MACOS_COLORS['bg_tertiary']};
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: {MACOS_COLORS['text_tertiary']};
            }}
            
            /* 加载动画 */
            @keyframes shimmer {{
                0% {{ background-position: -200% 0; }}
                100% {{ background-position: 200% 0; }}
            }}
            
            .loading {{
                background: linear-gradient(90deg, 
                    {MACOS_COLORS['bg_tertiary']} 25%, 
                    {MACOS_COLORS['bg_secondary']} 50%, 
                    {MACOS_COLORS['bg_tertiary']} 75%
                );
                background-size: 200% 100%;
                animation: shimmer 1.5s infinite;
            }}
            
            /* 脉冲动画 */
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
            
            .pulse {{
                animation: pulse 1.5s ease-in-out infinite;
            }}
        </style>
        """)
        
        def process_user_input(message, state, progress=gr.Progress()):
            if not message.strip():
                return "", "", conversation_display.value, f"""
                <div style="
                    text-align: center; 
                    padding: 30px; 
                    color: {MACOS_COLORS['accent_orange']};
                    background: {MACOS_COLORS['bg_secondary']};
                    border-radius: {MACOS_COLORS['radius_lg']};
                    border: 1px solid #FED7AA;
                ">
                    <div style="font-size: 36px; margin-bottom: 10px;">⚠️</div>
                    <p>请输入有效内容</p>
                </div>
                """, False, ""
            
            # 处理查询
            user_msg, search_result, updated_conversation, start_streaming = system.process_query(message, progress)
            
            system.current_streaming_response = ""
            
            return "", search_result, updated_conversation, search_result, True, ""
        
        def check_and_update_stream(state):
            if state and system.current_streaming_response:
                chunk, done = system.stream_response_step()
                if done:
                    new_state = False
                    system.memory.add_dialog("user", system.conversation_history[-1]['content'])
                    system.memory.add_dialog("assistant", system.current_streaming_response)
                    system.conversation_history.append({"role": "assistant", "content": system.current_streaming_response})
                    if len(system.conversation_history) > MAX_DIALOG_HISTORY * 2:
                        system.conversation_history = system.conversation_history[-MAX_DIALOG_HISTORY * 2:]
                    return system.current_streaming_response, system._get_conversation_history_html(), new_state
                elif chunk:
                    return system.current_streaming_response, system._get_conversation_history_html(), state
            return current_response.value, conversation_display.value, state
        
        def finish_streaming():
            return False, ""
        
        def update_status():
            return system.get_system_status_html()
        
        def clear_chat():
            msg, updated_conv = system.clear_conversation()
            new_status = update_status()
            gr.Info("对话历史已清空")
            return updated_conv, updated_conv, new_status, False, ""
        
        # 绑定事件
        submit_btn.click(
            fn=process_user_input,
            inputs=[user_input, streaming_state],
            outputs=[user_input, search_display, conversation_display, search_display, streaming_state, current_response]
        )
        
        user_input.submit(
            fn=process_user_input,
            inputs=[user_input, streaming_state],
            outputs=[user_input, search_display, conversation_display, search_display, streaming_state, current_response]
        )
        
        def check_and_update_stream_wrapper(state):
            result = check_and_update_stream(state)
            return result
        
        timer = gr.Timer(value=0.08)
        
        timer.tick(
            fn=check_and_update_stream_wrapper,
            inputs=[streaming_state],
            outputs=[current_response, conversation_display, streaming_state]
        )
        
        status_btn.click(
            fn=update_status,
            outputs=status_display
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[conversation_display, conversation_display, status_display, streaming_state, current_response]
        )
        
        demo.load(
            fn=update_status,
            outputs=status_display
        )
    
    return demo

# 简化版本 - 更稳定的流式实现
def create_interface_simple():
    system = GradioDialogSystem()
    
    # macOS风格主题
    macos_theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="green",
        neutral_hue="gray",
    ).set(
        body_background_fill="#F5F5F7",
        block_background_fill="#FFFFFF",
        block_border_width="1px",
        block_border_color="#E8E8ED",
        block_radius="14px",
        block_shadow="0 2px 12px rgba(0, 0, 0, 0.06)",
        input_background_fill="#FFFFFF",
        input_border_color="#D2D2D7",
        input_radius="10px",
        button_primary_background_fill="#007AFF",
        button_primary_background_fill_hover="#0062CC",
        button_primary_text_color="white",
        button_secondary_background_fill="#F5F5F7",
        button_secondary_background_fill_hover="#E8E8ED",
        button_secondary_text_color="#1D1D1F",
        color_accent_soft="#007AFF15",
    )
    
    with gr.Blocks(theme=macos_theme, title="基于信息-数据-知识三角理论的表征记忆系统demo") as demo:
        gr.Markdown(f"""
        <div style="
            text-align: center; 
            padding: 35px 20px 45px;
            background: linear-gradient(180deg, #FFFFFF 0%, #F5F5F7 100%);
            border-radius: 0 0 28px 28px;
            margin: -20px 0px 35px -20px;
        ">
            <h1 style="
                font-size: 2.4em; 
                font-weight: 700; 
                color: {MACOS_COLORS['text_primary']};
                margin-bottom: 12px;
                letter-spacing: -0.02em;
            ">基于信息-数据-知识三角理论的表征记忆系统demo</h1>
            <p style="
                font-size: 1.08em; 
                color: {MACOS_COLORS['text_secondary']};
                font-weight: 400;
            ">基于向量中心的记忆模块 · 实现长期对话记忆管理</p>
        </div>
        """)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                # 聊天显示区域
                chatbot = gr.Chatbot(
                    label="",
                    height=450,
                    bubble_full_width=False,
                    avatar_images=(
                        os.path.join(current_dir, "images/emoji.png"),
                        os.path.join(current_dir, "images/cat.png")
                    ),
                    elem_id="chatbot"
                )
                
                # 输入区域
                with gr.Group():
                    with gr.Row():
                        msg = gr.Textbox(
                            lines=2,
                            placeholder="输入您的问题，按 Enter 或点击发送...",
                            label="",
                            scale=4,
                            container=False,
                        )
                        submit_btn = gr.Button(
                            "发送", 
                            variant="primary", 
                            scale=1,
                        )
                
                # 控制按钮
                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")
                    status_btn = gr.Button("🔄 刷新状态", variant="secondary")
            
            with gr.Column(scale=1):
                # 系统状态
                status_display = gr.HTML(
                    value=system.get_system_status_html(),
                    label="系统状态",
                )
                
                # 记忆检索结果
                search_display = gr.HTML(
                    label="记忆检索结果",
                    value=f"""
                    <div style="
                        text-align: center; 
                        padding: 50px 20px; 
                        color: {MACOS_COLORS['text_tertiary']};
                        background: {MACOS_COLORS['bg_secondary']};
                        border-radius: 16px;
                        border: 1px dashed {MACOS_COLORS['bg_tertiary']};
                    ">
                        <div style="font-size: 48px; margin-bottom: 16px; opacity: 0.5;">🔍</div>
                        <p style="font-size: 1.05em; color: {MACOS_COLORS['text_secondary']};">等待查询...</p>
                    </div>
                    """,
                )
        
        # 自定义样式
        gr.HTML(f"""
        <style>
            #chatbot .user, #chatbot .bot {{
                border-radius: 18px !important;
                padding: 12px 16px !important;
                margin: 4px 0 !important;
            }}
            
            #chatbot .user {{
                background: {MACOS_COLORS['gradient_blue']} !important;
            }}
            
            #chatbot .bot {{
                background: {MACOS_COLORS['gradient_green']} !important;
            }}
            
            #chatbot .message {{
                color: white !important;
                font-size: 14.5px !important;
                line-height: 1.5 !important;
            }}
            
            .input-group textarea {{
                border-radius: 12px !important;
                border: 2px solid #E8E8ED !important;
                transition: all 0.2s ease !important;
            }}
            
            .input-group textarea:focus {{
                border-color: {MACOS_COLORS['accent_blue']} !important;
                box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.1) !important;
            }}
            
            .control-buttons {{
                margin-top: 8px;
            }}
            
            .control-buttons button {{
                font-size: 13px !important;
                transition: all 0.2s ease !important;
            }}
            
            .control-buttons button:hover {{
                transform: translateY(-1px) !important;
            }}
        </style>
        """)
        
        def respond(message, chat_history, progress=gr.Progress()):
            if not message.strip():
                return chat_history, f"""
                <div style="
                    text-align: center; 
                    padding: 35px; 
                    color: {MACOS_COLORS['accent_orange']};
                    background: {MACOS_COLORS['bg_secondary']};
                    border-radius: 16px;
                    border: 1px solid #FED7AA;
                ">
                    <div style="font-size: 40px; margin-bottom: 10px;">⚠️</div>
                    <p>请输入有效内容</p>
                </div>
                """, chat_history
            
            # 记忆检索
            progress(0.3, desc="正在检索记忆...")
            search_results = system.memory.search(message)
            search_html = system.build_pyramid_memory_display(search_results)
            
            # 生成流式响应
            progress(0.6, desc="AI正在思考...")
            system.conversation_history.append({"role": "user", "content": message})
            
            full_response = ""
            
            try:
                # 调用函数1生成流式响应
                for content, current_full in system.generate_response_stream(
                    prompt=message, 
                    context=search_results, 
                    progress=progress
                ):
                    if content is None:
                        # 流式响应结束
                        full_response = current_full
                        break
                    
                    # 更新响应
                    full_response = current_full
                    temp_history = chat_history + [(message, full_response)]
                    yield temp_history, search_html, temp_history
                
                # 保存对话
                system.memory.add_dialog("user", message)
                system.memory.add_dialog("assistant", full_response)
                system.conversation_history.append({"role": "assistant", "content": full_response})
                
                if len(system.conversation_history) > MAX_DIALOG_HISTORY * 2:
                    system.conversation_history = system.conversation_history[-MAX_DIALOG_HISTORY * 2:]
                
                yield chat_history + [(message, full_response)], search_html, chat_history + [(message, full_response)]
                    
            except Exception as e:
                error_msg = f"❌ 生成回复时出错: {str(e)}"
                yield chat_history + [(message, error_msg)], search_html, chat_history + [(message, error_msg)]
        
        msg.submit(
            respond,
            [msg, chatbot],
            [chatbot, search_display, chatbot]
        )
        
        submit_btn.click(
            respond,
            [msg, chatbot],
            [chatbot, search_display, chatbot]
        )
        
        def clear_chat():
            system.conversation_history = []
            new_status = system.get_system_status_html()
            gr.Info("对话历史已清空")
            empty_html = f"""
            <div style="
                text-align: center; 
                padding: 50px 20px; 
                color: {MACOS_COLORS['text_tertiary']};
                background: {MACOS_COLORS['bg_secondary']};
                border-radius: 16px;
                border: 1px dashed {MACOS_COLORS['bg_tertiary']};
            ">
                <div style="font-size: 48px; margin-bottom: 16px; opacity: 0.5;">🔍</div>
                <p style="font-size: 1.05em; color: {MACOS_COLORS['text_secondary']};">等待查询...</p>
            </div>
            """
            return [], empty_html, [], new_status
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, search_display, chatbot, status_display]
        )
        
        def refresh_status():
            return system.get_system_status_html()
        
        status_btn.click(
            refresh_status,
            outputs=status_display
        )
        
        demo.load(fn=lambda: system.get_system_status_html(), outputs=status_display)
    
    return demo

if __name__ == "__main__":
    print("🚀 启动智能记忆对话系统...")
    print("📱 访问地址: http://localhost:7860")
    demo = create_interface_simple()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
    )
