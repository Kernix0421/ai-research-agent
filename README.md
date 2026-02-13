# AI 科研文献问答系统

基于 RAG + BGE-M3 + Qwen 的本地化智能文献分析工具。

## 🚀 使用方法
1. 安装依赖：`pip install -r requirements.txt`
2. 配置 `.env`（参考 `.env.example`）
3. 构建索引：`python ingest.py`
4. 运行：`streamlit run app.py`

> 注意：向量库和日志不会上传，确保隐私安全。