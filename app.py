import streamlit as st
import time
import os
from dotenv import load_dotenv
load_dotenv()
import json
import re
import pandas as pd
from datetime import datetime
from models import qwen, deepseek, qwen_local
from ask import ResearchOrchestrator, WritingAgent, EvaluatorAgent


# --- 1. 基础配置与路径 ---
st.set_page_config(page_title="轨道交通科研助手 V2.2", page_icon="🔬", layout="wide")
CHAT_HISTORY_DB = os.getenv("CHAT_HISTORY_PATH", "./logs/full_chat_history.json")
LOG_PATH = os.getenv("RESEARCH_LOG_PATH", "./logs/research_agents_log.json")

# --- 2. 持久化逻辑函数 ---
def save_history_to_disk(messages):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    with open(CHAT_HISTORY_DB, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def load_history_from_disk():
    if os.path.exists(CHAT_HISTORY_DB):
        try:
            with open(CHAT_HISTORY_DB, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


# --- 3. 工具函数 ---
def WritingAgent_work_proxy(orchestrator, query, context, t_type, m_idx, limit):
    # 模型路由映射
    model_map = {"3": deepseek, "2": qwen, "4": qwen_local}
    llm = model_map.get(m_idx, qwen)
    writer = WritingAgent(llm, max_tokens=limit)
    return writer.work(query, context, t_type)


def format_all_history(messages):
    full_md = "# 全量科研对话历史\n\n"
    for m in messages:
        role = "👨‍🔬 研究员" if m["role"] == "user" else "🤖 AI 助手"
        full_md += f"### {role}\n{m['content']}\n\n---\n"
    return full_md


# --- 4. 初始化后端 ---
@st.cache_resource
def init_system():
    return ResearchOrchestrator()


boss = init_system()

# 初始化 Session State 并加载历史记录
if "messages" not in st.session_state:
    st.session_state.messages = load_history_from_disk()

# --- 5. 侧边栏控制面板 ---
with st.sidebar:
    st.title("⚙️ 系统控制台")
    task_type = st.radio("任务模式", ("1. 深度科研问答", "2. 项目申请辅助"))
    model_choice = st.selectbox(
        "选择写手引擎",
        ("3. DeepSeek-V3 (云端)", "2. Qwen-Plus (云端)", "4. Qwen3-VL-8B (本地)"),
        index=0
    )
    token_limit = st.slider("生成 Token 上限", 200, 4000, 1500)

    st.markdown("---")
    st.subheader("💾 数据管理")

    # 全量导出
    if st.session_state.messages:
        st.download_button(
            "📁 导出全量对话历史 (.md)",
            data=format_all_history(st.session_state.messages),
            file_name=f"Full_History_{datetime.now().strftime('%m%d_%H%M')}.md"
        )

    if st.button("🗑️ 清空所有历史", help="将删除本地存档文件"):
        st.session_state.messages = []
        if os.path.exists(CHAT_HISTORY_DB): os.remove(CHAT_HISTORY_DB)
        st.rerun()

    st.markdown("---")
    st.subheader("🧠 记忆管理")
    
    # 查看长期记忆
    if hasattr(boss, 'memory_manager') and boss.memory_manager.long_term_memory:
        st.write(f"长期记忆条数: {len(boss.memory_manager.long_term_memory)}")
        
        # 显示最近的记忆
        recent_memories = boss.memory_manager.long_term_memory[-5:]
        for i, memory in enumerate(reversed(recent_memories)):
            with st.expander(f"记忆 {i+1}: {memory['key']}"):
                st.write(memory['content'])
                st.caption(f"重要性: {memory['importance']}")
                st.caption(f"时间: {datetime.fromtimestamp(memory['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.write("暂无长期记忆")

    st.markdown("---")
    st.subheader("📈 最近任务评估")
    if os.path.exists(LOG_PATH):
        try:
            logs = json.load(open(LOG_PATH, "r", encoding="utf-8"))
            if logs:
                # 显示最近10条记录
                df = pd.DataFrame(logs).tail(10)[["timestamp", "query", "score", "duration"]]
                # 添加状态列
                df['状态'] = df['score'].apply(lambda x: '✅ 优秀' if x >= 80 else '⚠️ 良好' if x >= 60 else '❌ 需要改进')
                st.dataframe(df, hide_index=True)
        except:
            st.caption("日志暂不可用")

# --- 6. 主交互界面 ---
st.title("🔬 轨道交通科研增强系统 (Agentic V2.2)")
st.info("系统已就绪。所有对话将实时保存，您可以随时关闭并重新打开。")

# 渲染历史气泡
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if query := st.chat_input("输入科研问题..."):
    # 记录用户消息
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        start_time = time.time()
        status = st.status("🚀 Agent 团队协作中...", expanded=True)

        try:
            # A. 检索相关记忆
            memory_context = ""
            if hasattr(boss, 'memory_manager') and boss.memory_manager:
                status.write("🧠 记忆检索：正在获取相关历史知识...")
                try:
                    relevant_memory = boss.memory_manager.get_relevant_memory(query, max_length=800)
                    short_term_summary = boss.memory_manager.get_short_term_summary(max_length=400)
                    memory_context = f"{short_term_summary}\n\n{relevant_memory}"
                except Exception as e:
                    status.write(f"⚠️ 记忆检索失败: {e}")

            # B. 文档检索
            context = ""
            if hasattr(boss, 'researcher') and boss.researcher:
                status.write("🔍 资料员：正在进行向量与 BM25 混合检索...")
                try:
                    context, _ = boss.researcher.work(query, max_context_length=1500)
                except Exception as e:
                    status.write(f"⚠️ 文档检索失败: {e}")
                    context = "系统检索功能暂时不可用"
            else:
                status.write("⚠️ 检索功能不可用")
                context = "系统检索功能暂时不可用"
            
            full_context = f"{memory_context}\n\n{context}"

            # C. 写作
            m_idx = model_choice[0]
            status.write(f"✍️ 写手：调用 {model_choice[3:]} 撰写学术初稿...")
            try:
                draft, is_truncated = WritingAgent_work_proxy(boss, query, full_context, task_type[0], m_idx, token_limit)
            except Exception as e:
                draft = f"⚠️ 内容生成失败: {e}\n请稍后重试"
                is_truncated = False

            # D. 评估 (量化指标)
            status.write("⚖️ 评估员：正在进行多维指标分析...")
            try:
                evaluator = EvaluatorAgent(qwen)  # 固定用云端 Qwen 保证裁判权重
                eval_res = evaluator.work(query, draft, full_context)
                # 注意：此处 eval_res 需符合上个回复中定义的 JSON 格式
            except Exception as e:
                status.write(f"⚠️ 评估失败: {e}")
                eval_res = {"accuracy": 0, "precision": 0, "recall": 0, "hallucination": 0, "reason": "评估失败"}

            # E. 更新记忆
            if hasattr(boss, 'memory_manager') and boss.memory_manager:
                try:
                    boss.memory_manager.add_short_term(query, draft)
                    # 对于高评分内容，添加到长期记忆
                    if eval_res.get('accuracy', 0) > 80:
                        boss.memory_manager.add_long_term(f"重要知识点_{int(time.time())}", draft[:500])
                except Exception as e:
                    status.write(f"⚠️ 记忆更新失败: {e}")
        except Exception as e:
            status.write(f"❌ 系统执行失败: {e}")
            draft = f"⚠️ 系统执行失败: {e}\n请稍后重试"
            is_truncated = False
            eval_res = {"accuracy": 0, "precision": 0, "recall": 0, "hallucination": 0, "reason": "系统错误"}

        latency = time.time() - start_time
        status.update(label=f"✅ 任务完成 | 总耗时: {latency:.1f}s", state="complete")

        # --- 7. 指标展示看板 ---
        st.markdown("#### 📊 本轮科研质量指标")
        col1, col2, col3, col4 = st.columns(4)
        col5, col6, col7, col8 = st.columns(4)
        
        # 假设 eval_res 包含：accuracy, precision, recall, hallucination, relevance, completeness, innovation, clarity, reason
        col1.metric("准确率", f"{eval_res.get('accuracy', 0)}%")
        col2.metric("精确度", f"{eval_res.get('precision', 0)}%")
        col3.metric("召回率", f"{eval_res.get('recall', 0)}%")
        col4.metric("幻觉率", f"{eval_res.get('hallucination', 0)}%", delta_color="inverse")
        col5.metric("相关性", f"{eval_res.get('relevance', 0)}%")
        col6.metric("完整性", f"{eval_res.get('completeness', 0)}%")
        col7.metric("创新性", f"{eval_res.get('innovation', 0)}%")
        col8.metric("清晰度", f"{eval_res.get('clarity', 0)}%")
        
        # 耗时单独一行
        st.metric("总耗时", f"{latency:.1f}s")

        with st.expander("📝 查看评估理由"):
            st.write(eval_res.get('reason', '暂无详细分析'))

        # --- 8. 生成报告与导出 ---
        st.markdown("---")
        st.markdown(draft)

        report_md = f"# 科研报告: {query}\n\n**评估分: {eval_res.get('accuracy')}**\n\n{draft}\n\n---\n**评估意见:** {eval_res.get('reason')}"
        st.download_button(
            "📥 导出本条报告 (.md)",
            data=report_md,
            file_name=f"Report_{datetime.now().strftime('%m%d_%H%M')}.md"
        )

        if is_truncated: st.warning("⚠️ 内容已达到设定的 Token 上限，部分内容被截断。")

    # 记录 AI 消息并持久化
    st.session_state.messages.append({"role": "assistant", "content": draft})
    save_history_to_disk(st.session_state.messages)