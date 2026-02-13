import os
# 强制使用国内镜像站，提速并防止下载中断
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 解决可能的库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
from datetime import datetime
from pylatexenc.latex2text import LatexNodes2Text
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from unstructured.partition.auto import partition

# --- 核心配置 ---
DATA_PATH = os.getenv("LITERATURE_DATA_PATH", "./data/literature")
DB_PATH = os.getenv("FAISS_INDEX_PATH", "./vectorstore")
BM25_PATH = os.getenv("BM25_PICKLE_PATH", "./bm25_data.pkl")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# 初始化组件
print(f"⌛ 正在加载嵌入模型 {EMBED_MODEL}...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
latex_converter = LatexNodes2Text()

# 科研级切分器：512 token 大小，100 token 重叠
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "；", " ", ""]
)


def run_ingest():
    if not os.path.exists(DATA_PATH):
        print("❌ 错误：找不到文献路径")
        return

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)

    all_chunks = []
    files = [f for f in os.listdir(DATA_PATH) if f.endswith((".pdf", ".docx"))]

    for file in files:
        file_path = os.path.join(DATA_PATH, file)
        print(f"📄 解析中 (双栏/Fast模式): {file}")

        try:
            # strategy="fast" 绕过 Tesseract，利用 PDF 坐标处理双栏
            elements = partition(
                filename=file_path,
                strategy="fast",
                multipage_sections=True,
                chunking_strategy="by_title",  # 语义感知切分
                languages=["chi_sim", "eng"]
            )

            for el in elements:
                content = el.text
                # LaTeX 清洗
                if "$" in content or "\\" in content:
                    try:
                        content = latex_converter.latex_to_text(content)
                    except:
                        pass

                # 二次物理切分
                splits = text_splitter.split_text(content)
                for s in splits:
                    if len(s.strip()) < 15: continue
                    all_chunks.append(Document(
                        page_content=s,
                        metadata={
                            "source": file,
                            "type": el.category,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "project": "轨道技术研究"
                        }
                    ))
        except Exception as e:
            print(f"⚠️ 跳过文件 {file}: {e}")

    # 保存双索引：FAISS(语义) + BM25(关键词)
    if all_chunks:
        print(f"📦 构建索引中 (共 {len(all_chunks)} 个切片)...")
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        vectorstore.save_local(DB_PATH)
        with open(BM25_PATH, "wb") as f:
            pickle.dump(all_chunks, f)
        print("✅ 数据库构建完成！")


if __name__ == "__main__":
    run_ingest()