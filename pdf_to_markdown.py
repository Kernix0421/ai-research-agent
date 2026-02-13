#导入对应的库
import pymupdf4llm
import pathlib

#将pdf文档提取成markdown格式
md_text = pymupdf4llm.to_markdown(r'C:\Users\1\Desktop\研究方向\文献\2024铁路年报.pdf')
print(md_text)

#将提取出来的markdown文档，存储起来，存储为UTF-8的编码文件写入到md文件中
pathlib.Path("铁路年报.md").write_bytes(md_text.encode())

#返回成 LlamaIndex 文档
llama_reader = pymupdf4llm.LlamaMarkdownReader()
llama_docs = llama_reader.load_data(r'C:\Users\1\Desktop\研究方向\文献\2024铁路年报.pdf')