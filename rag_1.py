import os
import torch
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. 加载本地 HTML 文档
print("Current working directory:", os.getcwd())
with open("../data/llm_powered_autonomous_agents.html", "r", encoding="utf-8") as f:
    html_content = f.read()
soup = BeautifulSoup(html_content, "html.parser")
content = soup.find_all(class_=["post-content", "post-content-inner"])
docs = "\n".join([c.get_text() for c in content])
print("Loaded document content.")

# 2. 文本切分
doc_list = [Document(page_content=docs)]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
splits = text_splitter.split_documents(doc_list)
print(f"Split into {len(splits)} chunks.")

# 3. 构建向量数据库
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", 
)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory="../data/chroma_db",
)
retriever = vectorstore.as_retriever()
print("Vectorstore built and retriever ready.")

# 4. 加载本地 LLM (Gemma-3-4b-it)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    device_map="auto",
    torch_dtype="auto",
)
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# 5. 检索相关文档并生成答案
def format_docs(doc_list):
    return "\n\n".join(doc.page_content for doc in doc_list)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:
"""

context = format_docs(retriever.get_relevant_documents("what is Task Decomposition?"))
question = "what is Task Decomposition?"

prompt_text = template.format(context=context, question=question)
result = llm(prompt_text, max_new_tokens=500, temperature=0.1)
print(result[0]['generated_text'])

