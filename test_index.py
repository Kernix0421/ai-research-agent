import os
import sys

# 添加当前目录到路径
sys.path.append('.')

print("测试索引加载...")

# 测试1: 导入必要的模块
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever
    print("✅ 成功导入所有必要的模块")
except Exception as e:
    print(f"❌ 导入模块失败: {e}")
    sys.exit(1)

# 测试2: 加载嵌入模型
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"}
    )
    print("✅ 成功加载嵌入模型")
except Exception as e:
    print(f"❌ 加载嵌入模型失败: {e}")
    sys.exit(1)

# 测试3: 加载FAISS索引
DB_PATH = "./vectorstore"
try:
    vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    print("✅ 成功加载FAISS索引")
    # 测试检索
    test_query = "Kriging代理模型"
    results = vectorstore.similarity_search(test_query, k=3)
    print(f"✅ 成功执行检索，找到 {len(results)} 个结果")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.page_content[:100]}...")
except Exception as e:
    print(f"❌ 加载FAISS索引失败: {e}")

# 测试4: 加载BM25数据
BM25_PATH = "./bm25_data.pkl"
try:
    import pickle
    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = BM25Retriever.from_documents(bm25_data)
    print("✅ 成功加载BM25数据")
    # 测试检索
    test_query = "铁路选线"
    results = bm25.invoke(test_query)
    print(f"✅ 成功执行BM25检索，找到 {len(results)} 个结果")
    for i, result in enumerate(results[:3]):
        print(f"  {i+1}. {result.page_content[:100]}...")
except Exception as e:
    print(f"❌ 加载BM25数据失败: {e}")

# 测试5: 创建EnsembleRetriever
try:
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[0.6, 0.4])
    print("✅ 成功创建EnsembleRetriever")
    # 测试检索
    test_query = "代理模型优化"
    results = ensemble_retriever.invoke(test_query)
    print(f"✅ 成功执行混合检索，找到 {len(results)} 个结果")
    for i, result in enumerate(results[:3]):
        print(f"  {i+1}. {result.page_content[:100]}...")
except Exception as e:
    print(f"❌ 创建EnsembleRetriever失败: {e}")

print("\n测试完成！")
