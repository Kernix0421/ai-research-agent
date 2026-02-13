import os
from dotenv import load_dotenv
load_dotenv()
import re
import time
import json
import pickle
from datetime import datetime
from models import qwen, deepseek

# --- [1. 环境配置] ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "D:/AI_project/hf_cache"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
# --- 修正后的导入部分 (D:/AI_project/ask.py) ---

# 彻底放弃自动搜索，手动尝试所有可能的物理路径
import langchain
try:
    # 路径 1: 现代版本标准路径
    from langchain.retrievers.ensemble_retriever import EnsembleRetriever
except ImportError:
    try:
        # 路径 2: 某些 0.3.x 的变体路径
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        try:
            # 路径 3: 强制从模块根目录导入
            import importlib
            mod = importlib.import_module("langchain.retrievers.ensemble_retriever")
            EnsembleRetriever = mod.EnsembleRetriever
        except Exception as e:
            st.error(f"严重错误：无法定位检索组件。请检查安装。错误信息: {e}")

# 确保其他检索组件也正常
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# 路径常量
DB_PATH = os.getenv("FAISS_INDEX_PATH", "./vectorstore")
BM25_PATH = os.getenv("BM25_PICKLE_PATH", "./bm25_data.pkl")
LOG_FILE = os.getenv("RESEARCH_LOG_PATH", "./logs/research_agents_log.json")


# --- [2. 历史日志专家] ---
class HistoryLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        log_dir = os.path.dirname(filepath)
        if log_dir:  # 避免 filepath 是纯文件名（如 "log.json"）时出错
            os.makedirs(log_dir, exist_ok=True)

        # 如果日志文件不存在，初始化为空列表
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def log(self, data):
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            logs.append({**{"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, **data})
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 日志写入失败: {e}")


# --- [3. Agent 协作小组定义] ---

class ResearchAgent:
    """资料员：负责混合检索与证据清洗"""

    def __init__(self, retriever):
        self.retriever = retriever

    def work(self, query):
        print("🔍 [资料员] 检索中...")
        docs = self.retriever.invoke(query)
        context = "\n\n".join([f"【来源:{os.path.basename(d.metadata['source'])}】\n{d.page_content}" for d in docs])
        return context, docs


class WritingAgent:
    """写手：负责内容生成与 Token 控制"""

    def __init__(self, llm, max_tokens):
        self.llm = llm
        self.max_tokens = max_tokens

    def work(self, query, context, history, task_type):
        print(f"✍️ [写手] 撰写中 (限额:{self.max_tokens})...")

        mode_str = "科研问答" if task_type == "1" else "项目申请辅助"
        prompt = f"""你是一名{mode_str}专家。请基于证据回答，严禁幻觉。

【历史记忆】：{history}
【事实证据】：{context}
【当前问题】：{query}

要求：分点叙述，每条结论后加(来源:文件名)。"""

        # 绑定 Token 限制
        limited_llm = self.llm.bind(max_tokens=self.max_tokens)
        response = limited_llm.invoke(prompt)

        content = response.content
        finish_reason = response.response_metadata.get("finish_reason", "")
        return content, (finish_reason == "length")


class EvaluatorAgent:
    """升级版评估员：量化科研指标"""

    def __init__(self, llm):
        self.llm = llm

    def work(self, query, draft, context):
        prompt = f"""你是一名严谨的科研评审。请对比【事实证据】与【模型回答】，进行量化打分。

【事实证据】：{context}
【模型回答】：{draft}

请严格按以下 JSON 格式输出结果（不要包含其他文字）：
{{
  "accuracy": 0-100,      // 回答与证据的契合度
  "precision": 0-100,     // 回答中有效信息占比
  "recall": 0-100,        // 证据中关键点被采纳的比例
  "hallucination": 0-100, // 幻觉率（证据中未提及内容的比例）
  "reason": "简短的量化分析理由"
}}
"""
        # 强制使用 JSON 模式
        response = self.llm.bind(max_tokens=500).invoke(prompt)
        try:
            # 提取并解析 JSON
            import json
            res = json.loads(re.search(r'\{.*\}', response.content, re.S).group())
            return res
        except:
            # 保底方案
            return {"accuracy": 0, "precision": 0, "recall": 0, "hallucination": 0, "reason": "解析失败"}


# --- [4. 调度中心 (Orchestrator)] ---

class ResearchOrchestrator:
    def __init__(self):
        self.logger = HistoryLogger(LOG_FILE)
        self.retriever = self._init_retriever()
        self.researcher = ResearchAgent(self.retriever)
        self.chat_memory = ""  # 简单记忆

    def _init_retriever(self):
        print("⌛ [初始化] 加载 Embedding 与索引...")
        emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': 'cpu'})
        vs = FAISS.load_local(DB_PATH, emb, allow_dangerous_deserialization=True)
        with open(BM25_PATH, "rb") as f: bm25_data = pickle.load(f)
        bm25 = BM25Retriever.from_documents(bm25_data)
        bm25.k = 4
        return EnsembleRetriever(retrievers=[vs.as_retriever(search_kwargs={"k": 7}), bm25], weights=[0.7, 0.3])

    def execute(self, query, task_type, model_choice, t_limit):
        # 模型分配：写手(DeepSeek逻辑强), 评估员(Qwen指令强)
        writer_llm = deepseek if model_choice == "3" else qwen
        eval_llm = qwen

        writer = WritingAgent(writer_llm, max_tokens=t_limit)
        evaluator = EvaluatorAgent(eval_llm)

        start_time = time.time()

        # 1. 检索
        context, raw_docs = self.researcher.work(query)

        # 2. 写作
        draft, is_truncated = writer.work(query, context, self.chat_memory, task_type)
        if is_truncated: draft = "⚠️(内容因长度截断)\n" + draft

        # 3. 评估
        score, feedback = evaluator.work(query, draft, context)

        # 4. 存入记忆与日志
        duration = f"{time.time() - start_time:.2f}s"
        self.chat_memory = f"Q:{query} A:{draft[:100]}..."  # 记忆压缩

        log_data = {
            "query": query, "score": score, "duration": duration,
            "model": "DeepSeek-V3" if model_choice == "3" else "Qwen-Plus",
            "is_truncated": is_truncated
        }
        self.logger.log(log_data)

        return draft, feedback, score, duration


# --- [5. 交互界面] ---

def main():
    print("\n" + "=" * 60)
    print("🔬 AI科研助手 Agent V1.5 | 自动存档 | Token保护 | 多Agent互审")
    print("=" * 60)

    boss = ResearchOrchestrator()

    while True:
        query = input("\n[提问] (q退出): ")
        if query.lower() == 'q': break

        print("任务: 1.科研问答 2.项目申请 | 模型: 2.Qwen 3.DeepSeek")
        t_type = input("任务(默认1): ") or "1"
        m_type = input("模型(默认3): ") or "3"
        t_limit = int(input("Token上限(默认1000): ") or "1000")

        draft, feedback, score, cost = boss.execute(query, t_type, m_type, t_limit)

        print("\n" + "—" * 20 + " ✍️ 写手报告 " + "—" * 20)
        print(draft)
        print("\n" + "—" * 20 + " ⚖️ 评估意见 (得分:{}) ".format(score) + "—" * 20)
        print(feedback)
        print(f"\n📊 统计: 耗时{cost} | 日志已同步至 JSON")


if __name__ == "__main__":
    main()