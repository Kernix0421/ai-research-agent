import os
import sys

# 添加当前目录到路径
sys.path.append('.')

print("测试检索功能...")

# 测试1: 导入EnsembleRetriever
try:
    from langchain_classic.retrievers import EnsembleRetriever
    print("✅ 成功导入 EnsembleRetriever from langchain_classic.retrievers")
except ImportError as e:
    print(f"❌ 导入失败: {e}")

# 测试2: 检查其他依赖
print("\n检查其他依赖...")
try:
    from langchain_community.vectorstores import FAISS
    print("✅ 成功导入 FAISS")
except ImportError as e:
    print(f"❌ FAISS 导入失败: {e}")

try:
    from langchain_community.retrievers import BM25Retriever
    print("✅ 成功导入 BM25Retriever")
except ImportError as e:
    print(f"❌ BM25Retriever 导入失败: {e}")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ 成功导入 HuggingFaceEmbeddings")
except ImportError as e:
    print(f"❌ HuggingFaceEmbeddings 导入失败: {e}")

# 测试3: 尝试初始化嵌入模型
print("\n尝试初始化嵌入模型...")
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    # 使用一个较小的模型，避免下载大模型
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    print("✅ 成功初始化嵌入模型")
except Exception as e:
    print(f"❌ 初始化嵌入模型失败: {e}")

# 测试4: 检查索引文件
print("\n检查索引文件...")
vectorstore_dir = "./vectorstore"
bm25_path = "./bm25_data.pkl"

if os.path.exists(vectorstore_dir):
    print(f"✅ 向量存储目录存在: {vectorstore_dir}")
    files = os.listdir(vectorstore_dir)
    print(f"   目录内容: {files}")
else:
    print(f"❌ 向量存储目录不存在: {vectorstore_dir}")

if os.path.exists(bm25_path):
    print(f"✅ BM25数据文件存在: {bm25_path}")
else:
    print(f"❌ BM25数据文件不存在: {bm25_path}")

print("\n测试完成！")
