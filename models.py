import os
from langchain_openai import ChatOpenAI

api_key = os.getenv("DASHSCOPE_API_KEY")

# --- 云端模型 ---
qwen = ChatOpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.1
)

deepseek = ChatOpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="deepseek-v3",
    temperature=0.1
)

# --- 本地模型 (Qwen2-VL-8B) ---
# 假设您本地使用 Ollama 或 vLLM 开启了 OpenAI 兼容接口
qwen_local = ChatOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:11434/v1", # 根据您的实际本地端口修改
    model="qwen3-vl:8b",                  # 对应您本地加载的模型名称
    temperature=0.1
)