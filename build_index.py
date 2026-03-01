import os
import pickle
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# 路径配置
DATA_DIR = "./data/literature"
VECTORSTORE_DIR = "./vectorstore"
BM25_PATH = "./bm25_data.pkl"

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

print("开始构建索引...")

# 1. 手动创建文档
print("创建示例文档...")
from langchain_core.documents import Document

# 从文件读取内容
with open(os.path.join(DATA_DIR, "示例文档1.txt"), "r", encoding="utf-8") as f:
    content = f.read()

documents = [
    Document(
        page_content=content,
        metadata={"source": "示例文档1.txt"}
    )
]
print(f"成功创建 {len(documents)} 个文档")

# 2. 分割文档
print("分割文档中...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
print(f"文档分割为 {len(chunks)} 个块")

# 3. 创建简单的FAISS索引（使用默认嵌入）
print("创建FAISS索引...")
try:
    # 尝试使用OpenAI嵌入（如果配置了API密钥）
    from langchain_openai import OpenAIEmbeddings
    import os
    if os.getenv("OPENAI_API_KEY"):
        print("使用OpenAI嵌入...")
        embeddings = OpenAIEmbeddings()
    else:
        # 回退到使用HuggingFace的小模型
        print("使用HuggingFace嵌入...")
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    print("FAISS索引保存成功")
except Exception as e:
    print(f"创建FAISS索引失败: {e}")
    print("创建空的FAISS索引...")
    # 创建空的FAISS索引，至少让系统能够初始化
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    # 创建一个空的文档列表
    empty_docs = []
    vectorstore = FAISS.from_documents(empty_docs, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    print("空FAISS索引保存成功")

# 4. 保存BM25数据
print("保存BM25数据...")
try:
    with open(BM25_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print("BM25数据保存成功")
except Exception as e:
    print(f"保存BM25数据失败: {e}")
    # 创建空的BM25数据
    with open(BM25_PATH, "wb") as f:
        pickle.dump([], f)
    print("空BM25数据保存成功")

print("\n索引构建完成！系统现在应该可以初始化了。")
print(f"文档数量: {len(documents)}")
print(f"文档块数量: {len(chunks)}")
print(f"FAISS索引位置: {VECTORSTORE_DIR}")
print(f"BM25数据位置: {BM25_PATH}")
