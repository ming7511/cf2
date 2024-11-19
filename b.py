import os
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy as np


# 1. 文档解析：解析PDF文档为文本数据
def parse_pdfs_in_folder(folder_path):
    texts = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    texts.append(page.extract_text())
    return "\n".join(texts)


# 2. 文本分块
def split_text_into_chunks(text, chunk_size=200):
    sentences = text.split('\n')
    chunks, current_chunk = [], []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) <= chunk_size:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# 3. 向量化文本块
def generate_embeddings(text_chunks, model):
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    return embeddings


# 4. 向量检索：基于Milvus的检索函数
def search_query(question, model, collection, top_k=5):
    # 将问题向量化
    query_embedding = model.encode([question], convert_to_numpy=True)

    # 搜索向量数据库
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k
    )

    # 提取文本块和向量
    matches = []
    for result in results[0]:
        matches.append({
            "text": result.entity.get("text"),
            "embedding": result.entity.get("embedding"),
            "distance": result.distance
        })
    return matches


# 5. 提交文件格式化
def generate_submission_file(questions, results, output_file="submit_example.csv"):
    data = []
    for ques_id, question, matches in zip(questions["ques_id"], questions["question"], results):
        best_match = matches[0]  # 假设选择距离最近的文本块
        data.append({
            "ques_id": ques_id,
            "question": question,
            "answer": best_match["text"],
            "embedding": " ".join(map(str, best_match["embedding"]))
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding="utf-8")


# 主流程
if __name__ == "__main__":
    # 加载问题文件
    questions_file_path = "./b榜/B_question.csv"
    questions_df = pd.read_csv(questions_file_path)

    # 第1步：文档解析
    folder_path = "./b榜/B_document"  # 文档文件夹路径
    pdf_text = parse_pdfs_in_folder(folder_path)
    print("Parsed Text:", pdf_text[:500])  # 打印部分解析内容

    # 第2步：加载bge-large-zh-v1.5模型
    model = SentenceTransformer("./model")  # 模型路径

    # 第3步：文本分块
    chunks = split_text_into_chunks(pdf_text)
    print("Number of Chunks:", len(chunks))

    # 第4步：生成向量
    embeddings = generate_embeddings(chunks, model)

    # 第5步：向量存储入Milvus
    connections.connect("default", host="127.0.0.1", port="19530")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embeddings.shape[1]),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
    ]
    schema = CollectionSchema(fields, "Text embedding storage")
    collection = Collection("knowledge_base", schema)
    collection.insert([list(range(len(chunks))), embeddings.tolist(), chunks])
    collection.load()
    print(f"Inserted {len(chunks)} chunks into vector database.")

    # 第6步：文本召回
    results = [search_query(q, model, collection) for q in questions_df["question"]]

    # 第7步：生成提交文件
    generate_submission_file(questions_df, results)
    print("Submission file generated: submit_example.csv")
