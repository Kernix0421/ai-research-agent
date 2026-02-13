#提供跨平台的操作系统交互接口，允许开发者执行文件管理、进程控制、环境配置等底层操作
import os
#专业日志管理核心工具，可以监控生产环境、调试、运行状态等
import logging
#Python中的序列化库，可以将Python对象序列化二进制字节流，保存到文件在中，同时可以反序列化还原对象
import pickle
#用于读取PDF文件内容，提取文本、元数据（作者/标题）、页面对象等基础信息
from PyPDF2 import PdfReader
#加载预定义的问答任务链，自动化实现“检索→生成”流程
from langchain_classic.chains.question_answering import load_qa_chain
#调用OpenAI的GPT系列模型,实现文本生成、问答、摘要等任务
from langchain_openai import OpenAI, ChatOpenAI
#阿里云DashScope提供的嵌入模型，适合中文场景优化，功能类似OpenAI Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
#监控OpenAI API调用开销（消耗的tokens数量、费用），调试性能瓶颈
from langchain_community.callbacks.manager import get_openai_callback
#将长文本按语义递归分割为小片段（如按段落、句子），适配语言模型的上下文长度限制
from langchain_text_splitters import RecursiveCharacterTextSplitter
#Facebook开源的向量数据库库，支持快速相似性搜索（最近邻检索）
from langchain_community.vectorstores import FAISS
#Python类型标注库，用于声明函数参数/返回值的类型（如List[str]表示字符串列表），提升代码可读性与静态检查支持
from typing import List, Tuple

#函数一：读取pdf文件，提取每行内容和每行对应的页码，同时拼接成一个大文本
def extract_text_with_page_numbers(pdf) -> Tuple[str,List[int]]:
    """
    从pdf中提取文本并记录每行文本对应的页码

    参数：
        pdf: PDF文件对象

    返回:
        text: 提取的文本内容
        page_numbers: 每行文本对应的页码列表
    """

    text = ""
    page_numbers = []

    #遍历PDF的每一页，enumerate从1开始计数页码
    for page_number,page in enumerate(pdf.pages, start=1):
        #提取当前页的文本
        extracted_text = page.extract_text()
        #如果该页面有文本
        if extracted_text:
            #将当前页文本追加到总文本中（字符串拼接文本）
            text += extracted_text
            #记录每行文本对应的页码（按换行符分隔）
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))
        else:
            #处理无文本页面的情况
            logging.warning(f"NO TEXT FOUND ON page {page_number}.")
    return text ,page_numbers

#向量化处理函数，对识别到的PDF文件进行分割，将分割好的文本块构建成向量存储
def process_text_with_splitter(text: str, page_numbers: List[int], save_path: str = None):
    """
       处理文本并创建向量存储

       参数:
           text: 提取的文本内容
           page_numbers: 每行文本对应的页码列表
           save_path: 可选，保存向量数据库的路径

       返回:
           knowledgeBase: 基于FAISS的向量存储对象
       """
    #创建递归式文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "\t", "."," "], #按语义单位分割
        chunk_size=512,  #区块最大长度
        chunk_overlap= 128,   #区块间重叠长度
        length_function= len,   #长度计算函数
    )

    #文本分块处理
    chunks = text_splitter.split_text(text)
    #print(f"文本被分割成: {len(chunks)}个块。")

    #使用阿里云DashScope生成文本嵌入向量
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    #构建FAISS向量数据库
    knowledgeBase = FAISS.from_texts(chunks,embeddings)
    #print("已从文本块创建知识库...")
    #建立区块-页码映射关系,enumerate(chunks)同时获取分块的索引 i和内容 chunk。
    #page_numbers[i]表示第 i个分块在源文档中的起始页码
    page_info = {chunk: page_numbers[i] for i, chunk in enumerate(chunks)}
    knowledgeBase.page_info = page_info   #附加元数据

    #持久化存储
    if save_path:
        #确保目录存在
        os.makedirs(save_path, exist_ok=True)
        # 保存向量索引
        knowledgeBase.save_local(save_path)
        #print(f"向量数据库已保存到: {save_path}")
        #序列化储存页码元数据，保存页码信息到同一目录
        with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
            pickle.dump(page_info, f)
        #print(f"页码信息已保存到: {os.path.join(save_path, 'page_info.pkl')}")

    return knowledgeBase

def load_knowledge_base(load_path: str, embeddings = None) -> FAISS:
    """
        从磁盘加载向量数据库和页码信息

        参数:
            load_path: 向量数据库的保存路径
            embeddings: 可选，嵌入模型。如果为None，将创建一个新的DashScopeEmbeddings实例

        返回:
            knowledgeBase: 加载的FAISS向量数据库对象
    """

    #初始化嵌入模型（默认阿里云）
    if embeddings is None:
        embeddings = DashScopeEmbeddings(model="text-embedding-v2")


    #加载FAISS向量库（需要启用反序列化安全选项）
    knowledgeBase = FAISS.load_local(
        load_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    #加载页码元数据
    page_info_path = os.path.join(load_path, "page_info.pkl")
    if os.path.exists(page_info_path):
        with open(page_info_path, "rb") as f:
            knowledgeBase.page_info = pickle.load(f)

    return knowledgeBase