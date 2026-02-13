from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# 设置目标目录路径
directory_path = r"C:\Users\1\Desktop\研究方向\文献"

# 创建 DirectoryLoader 实例

loader = DirectoryLoader(
path=directory_path,      # 目标目录路径
glob="**/*.pdf",          # 递归匹配所有 PDF 文件
loader_cls=PyPDFLoader,   # 指定 PDF 加载器
show_progress=True,       # 显示加载进度条（可选）
# use_multithreading=True   # 启用多线程加速（可选）
)

# 加载文档
documents = loader.load()
print(documents)