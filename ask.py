import os
from dotenv import load_dotenv
load_dotenv()
import re
import time
import json
import pickle
from datetime import datetime
from models import qwen, deepseek
import streamlit as st

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
EnsembleRetriever = None
try:
    # 路径 1: 现代版本标准路径 (LangChain 1.x) - langchain_classic
    from langchain_classic.retrievers import EnsembleRetriever
    print("✅ 成功导入 EnsembleRetriever from langchain_classic.retrievers")
except ImportError:
    try:
        # 路径 2: 现代版本标准路径 (LangChain 1.x) - langchain_community
        from langchain_community.retrievers import EnsembleRetriever
        print("✅ 成功导入 EnsembleRetriever from langchain_community.retrievers")
    except ImportError:
        try:
            # 路径 3: 旧版本路径
            from langchain.retrievers import EnsembleRetriever
            print("✅ 成功导入 EnsembleRetriever from langchain.retrievers")
        except ImportError:
            try:
                # 路径 4: 另一种可能的路径
                from langchain.retrievers.ensemble_retriever import EnsembleRetriever
                print("✅ 成功导入 EnsembleRetriever from langchain.retrievers.ensemble_retriever")
            except ImportError:
                try:
                    # 路径 5: 强制从模块根目录导入
                    import importlib
                    try:
                        mod = importlib.import_module("langchain_classic.retrievers")
                        EnsembleRetriever = mod.EnsembleRetriever
                        print("✅ 成功导入 EnsembleRetriever from langchain_classic.retrievers via importlib")
                    except:
                        try:
                            mod = importlib.import_module("langchain_community.retrievers")
                            EnsembleRetriever = mod.EnsembleRetriever
                            print("✅ 成功导入 EnsembleRetriever from langchain_community.retrievers via importlib")
                        except:
                            mod = importlib.import_module("langchain.retrievers.ensemble_retriever")
                            EnsembleRetriever = mod.EnsembleRetriever
                            print("✅ 成功导入 EnsembleRetriever from langchain.retrievers.ensemble_retriever via importlib")
                except Exception as e:
                    print(f"⚠️ 无法定位检索组件: {e}")
                    print("⚠️ 系统将以降级模式运行，检索功能暂时不可用")

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

    def work(self, query, max_context_length=2000):
        print("🔍 [资料员] 检索中...")
        
        # 查询扩展
        expanded_query = self._expand_query(query)
        print(f"📝 扩展查询: {expanded_query}")
        
        # 使用扩展后的查询进行检索
        try:
            docs = self.retriever.invoke(expanded_query)
        except Exception as e:
            print(f"⚠️ 检索失败: {e}")
            # 返回空结果
            return "系统检索功能暂时不可用", []
        
        # 处理文档，提取关键信息并压缩
        processed_docs = []
        total_length = 0
        
        for d in docs:
            try:
                # 提取关键信息
                content = self._extract_key_info(d.page_content)
                # 压缩文档内容
                compressed_content = self._compress_document(content, max_length=500)
                # 安全获取来源信息
                source = d.metadata.get('source', '未知来源')
                doc_str = f"【来源:{os.path.basename(source)}】\n{compressed_content}"
                
                # 控制总长度
                if total_length + len(doc_str) < max_context_length:
                    processed_docs.append(doc_str)
                    total_length += len(doc_str)
                else:
                    break
            except Exception as e:
                print(f"⚠️ 处理文档失败: {e}")
                continue
        
        context = "\n\n".join(processed_docs)
        if not context:
            context = "系统检索功能暂时不可用"
        return context, docs
    
    def _expand_query(self, query):
        """查询扩展"""
        # 简单的查询扩展：添加同义词和相关术语
        synonyms = {
            "轨道交通": ["地铁", "轻轨", "高铁", "铁路"],
            "人工智能": ["AI", "机器学习", "深度学习"],
            "优化": ["改进", "提升", "增强"],
            "模型": ["算法", "方法", "技术"],
            "研究": ["调研", "分析", "探索"]
        }
        
        expanded_terms = [query]
        
        # 提取关键词并添加同义词
        for term, syns in synonyms.items():
            if term in query:
                expanded_terms.extend(syns)
        
        # 去重并组合
        unique_terms = list(set(expanded_terms))
        return " ".join(unique_terms)
    
    def _extract_key_info(self, content):
        """提取关键信息"""
        # 简单的关键信息提取：保留标题和段落首句
        lines = content.split('\n')
        key_lines = []
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('#') or line.endswith('。') or line.endswith('!') or line.endswith('?')):
                key_lines.append(line)
        
        return '\n'.join(key_lines[:10])  # 最多保留10行关键信息
    
    def _compress_document(self, content, max_length=500):
        """压缩文档内容"""
        if len(content) <= max_length:
            return content
        # 简单压缩：保留开头和结尾，中间用省略号
        half = (max_length - 3) // 2
        return content[:half] + "..." + content[-half:]


class WritingAgent:
    """写手：负责内容生成与 Token 控制"""

    def __init__(self, llm, max_tokens):
        self.llm = llm
        self.max_tokens = max_tokens

    def work(self, query, context, task_type):
        print(f"✍️ [写手] 撰写中 (限额:{self.max_tokens})...")

        mode_str = "科研问答" if task_type == "1" else "项目申请辅助"
        prompt = f"""你是一名{mode_str}专家。请基于证据回答，严禁幻觉。

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
  "relevance": 0-100,     // 回答与查询的相关性
  "completeness": 0-100,  // 回答的完整性
  "innovation": 0-100,    // 回答的创新性
  "clarity": 0-100,       // 回答的清晰度
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
            return {
                "accuracy": 0, "precision": 0, "recall": 0, "hallucination": 0,
                "relevance": 0, "completeness": 0, "innovation": 0, "clarity": 0,
                "reason": "解析失败"
            }


# --- [4. 记忆管理系统] ---

class MemoryManager:
    """分层记忆管理器"""
    
    def __init__(self, embedding_model):
        self.short_term_memory = []  # 最近对话，最多保存10轮
        self.long_term_memory = []  # 重要信息，带向量嵌入
        self.embedding_model = embedding_model
        self._embedding_cache = {}  # 嵌入缓存，避免重复计算
    
    def _get_embedding(self, text):
        """获取文本嵌入，使用缓存"""
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.embedding_model.embed_query(text)
        return self._embedding_cache[text]
    
    def add_short_term(self, query, response):
        """添加短期记忆"""
        self.short_term_memory.append({"query": query, "response": response, "timestamp": time.time()})
        # 保持最近10轮对话
        if len(self.short_term_memory) > 10:
            self.short_term_memory = self.short_term_memory[-10:]
    
    def add_long_term(self, key, content, importance=0.8):
        """添加长期记忆"""
        # 压缩内容，保持在合理长度
        compressed_content = self._compress_content(content, max_length=800)
        embedding = self._get_embedding(compressed_content)
        self.long_term_memory.append({
            "key": key,
            "content": compressed_content,
            "importance": importance,
            "embedding": embedding,
            "timestamp": time.time()
        })
    
    def get_relevant_memory(self, query, k=3, max_length=1000):
        """获取与查询相关的记忆"""
        if not self.long_term_memory:
            return ""
        
        # 计算查询向量
        query_embedding = self._get_embedding(query)
        
        # 计算相似度
        import numpy as np
        similarities = []
        for memory in self.long_term_memory:
            similarity = np.dot(query_embedding, memory["embedding"]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(memory["embedding"]))
            # 结合重要性和相似度
            combined_score = similarity * 0.7 + memory["importance"] * 0.3
            similarities.append((combined_score, memory))
        
        # 按相似度排序，取前k个
        similarities.sort(reverse=True)
        relevant_memories = [m[1] for m in similarities[:k]]
        
        # 格式化并压缩返回
        memory_str = "\n\n".join([f"【记忆:{m['key']}】\n{m['content']}" for m in relevant_memories])
        return self._compress_content(memory_str, max_length)
    
    def get_short_term_summary(self, max_length=500):
        """获取短期记忆摘要"""
        if not self.short_term_memory:
            return ""
        
        # 生成摘要
        summary = "最近对话摘要：\n"
        for i, item in enumerate(reversed(self.short_term_memory)):
            summary += f"Q{i+1}: {item['query'][:50]}...\nA{i+1}: {item['response'][:50]}...\n"
        return self._compress_content(summary, max_length)
    
    def _compress_content(self, content, max_length=500):
        """压缩内容到指定长度"""
        if len(content) <= max_length:
            return content
        # 简单压缩：保留开头和结尾，中间用省略号
        half = (max_length - 3) // 2
        return content[:half] + "..." + content[-half:]

# --- [5. 调度中心 (Orchestrator)] ---

class ResearchOrchestrator:
    def __init__(self):
        try:
            self.logger = HistoryLogger(LOG_FILE)
            self.retriever = self._init_retriever()
            
            # 只有当retriever成功初始化时才创建ResearchAgent
            if self.retriever:
                self.researcher = ResearchAgent(self.retriever)
            else:
                self.researcher = None
                print("⚠️ 检索功能暂时不可用")
            
            # 尝试初始化记忆管理系统
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': 'cpu'})
                self.memory_manager = MemoryManager(self.embedding_model)
            except Exception as e:
                print(f"⚠️ 记忆管理系统初始化失败: {e}")
                self.memory_manager = None
            
            print("✅ 系统初始化成功")
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            # 初始化失败时的降级处理
            self.logger = HistoryLogger(LOG_FILE)
            self.researcher = None
            self.memory_manager = None

    def _init_retriever(self):
        print("⌛ [初始化] 加载 Embedding 与索引...")
        try:
            # 检查EnsembleRetriever是否可用
            if EnsembleRetriever is None:
                print("⚠️ EnsembleRetriever 不可用")
                # 尝试直接使用BM25
                try:
                    with open(BM25_PATH, "rb") as f: 
                        bm25_data = pickle.load(f)
                    bm25 = BM25Retriever.from_documents(bm25_data)
                    bm25.k = 10
                    print("✅ 成功初始化BM25检索器")
                    return bm25
                except Exception as e:
                    print(f"⚠️ 无法初始化BM25检索器: {e}")
                    return None
            
            # 尝试加载BM25数据
            try:
                with open(BM25_PATH, "rb") as f: 
                    bm25_data = pickle.load(f)
                bm25 = BM25Retriever.from_documents(bm25_data)
                print("✅ 成功初始化BM25检索器")
            except Exception as e:
                print(f"⚠️ 无法初始化BM25检索器: {e}")
                return None
            
            # 尝试初始化嵌入模型和FAISS索引
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                # 尝试使用本地嵌入模型
                emb = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"}
                )
                print("✅ 成功初始化嵌入模型")
                
                # 尝试加载FAISS索引
                vs = FAISS.load_local(DB_PATH, emb, allow_dangerous_deserialization=True)
                print("✅ 成功加载FAISS索引")
                
                # 优化参数
                bm25.k = 5  # 增加BM25检索结果数量
                vector_retriever = vs.as_retriever(search_kwargs={"k": 8})  # 增加向量检索结果数量
                
                # 调整权重，增加BM25的权重以提高关键词匹配效果
                retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[0.6, 0.4])
                print("✅ 成功初始化EnsembleRetriever")
                return retriever
            except Exception as e:
                print(f"⚠️ 无法初始化FAISS检索器: {e}")
                print("⚠️ 仅使用BM25检索器")
                # 仅使用BM25
                bm25.k = 10
                return bm25
        except Exception as e:
            print(f"⚠️ 检索器初始化失败: {e}")
            print("⚠️ 系统将以降级模式运行，检索功能暂时不可用")
            # 返回None，让系统以降级模式运行
            return None

    def execute(self, query, task_type, model_choice, t_limit):
        try:
            # 模型分配：写手(DeepSeek逻辑强), 评估员(Qwen指令强)
            writer_llm = deepseek if model_choice == "3" else qwen
            eval_llm = qwen

            writer = WritingAgent(writer_llm, max_tokens=t_limit)
            evaluator = EvaluatorAgent(eval_llm)

            start_time = time.time()

            # 1. 检索相关记忆
            memory_context = ""
            if self.memory_manager:
                try:
                    relevant_memory = self.memory_manager.get_relevant_memory(query, max_length=800)
                    short_term_summary = self.memory_manager.get_short_term_summary(max_length=400)
                    memory_context = f"{short_term_summary}\n\n{relevant_memory}"
                except Exception as e:
                    print(f"⚠️ 记忆检索失败: {e}")

            # 2. 检索文档
            context = ""
            if self.researcher:
                try:
                    context, raw_docs = self.researcher.work(query, max_context_length=1500)
                except Exception as e:
                    print(f"⚠️ 文档检索失败: {e}")
            else:
                context = "系统检索功能暂时不可用"
            
            full_context = f"{memory_context}\n\n{context}"

            # 3. 写作
            draft = ""
            is_truncated = False
            try:
                draft, is_truncated = writer.work(query, full_context, task_type)
                if is_truncated: draft = "⚠️(内容因长度截断)\n" + draft
            except Exception as e:
                draft = f"⚠️ 内容生成失败: {e}\n请稍后重试"

            # 4. 评估
            score = 0
            feedback = "评估失败"
            try:
                eval_result = evaluator.work(query, draft, full_context)
                score = eval_result.get('accuracy', 0)
                feedback = eval_result.get('reason', '评估失败')
            except Exception as e:
                print(f"⚠️ 评估失败: {e}")

            # 5. 更新记忆
            if self.memory_manager:
                try:
                    self.memory_manager.add_short_term(query, draft)
                    # 对于重要的内容，添加到长期记忆
                    if score > 80:
                        self.memory_manager.add_long_term(f"重要知识点_{int(time.time())}", draft[:500])
                except Exception as e:
                    print(f"⚠️ 记忆更新失败: {e}")

            # 6. 存入日志
            duration = f"{time.time() - start_time:.2f}s"

            log_data = {
                "query": query, "score": score, "duration": duration,
                "model": "DeepSeek-V3" if model_choice == "3" else "Qwen-Plus",
                "is_truncated": is_truncated
            }
            self.logger.log(log_data)

            return draft, feedback, score, duration
        except Exception as e:
            print(f"❌ 执行失败: {e}")
            # 降级处理
            return f"⚠️ 系统执行失败: {e}\n请稍后重试", "系统错误", 0, "0.00s"


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