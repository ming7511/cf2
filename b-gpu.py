import os
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
import torch
import re

# 确保 PyTorch 使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. 文档解析
def parse_pdfs_in_folder(folder_path):
    texts = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        texts.append(page_text)
    return "\n".join(texts)

# 2. 文本分块
def split_text_into_chunks(text, max_chunk_length=2048):
    sentences = re.split(r'(?<=[。！？])', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) <= max_chunk_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(sentence) > max_chunk_length:
                # 对于超长的句子，进行切分
                for i in range(0, len(sentence), max_chunk_length):
                    chunks.append(sentence[i:i+max_chunk_length])
                current_chunk = ""
            else:
                current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# 3. 向量化文本块
def generate_embeddings(text_chunks, model):
    embeddings = model.encode(text_chunks, device=device, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

# 4. 向量检索
def search_query(question, model, collection, top_k=5):
    query_embedding = model.encode([question], device=device, convert_to_numpy=True)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )
    matches = []
    for result in results[0]:
        matches.append({
            "text": result.entity.get("text"),
            "distance": result.distance
        })
    return matches

# 5. 提交文件格式化
def generate_submission_file(questions, results, model, output_file="submit_example.csv"):
    data = []
    for ques_id, question, matches in zip(questions["ques_id"], questions["question"], results):
        if matches:
            best_match = matches[0]
            answer_text = best_match["text"]
            answer_embedding = model.encode([answer_text], device=device, convert_to_numpy=True)[0]
            embedding_str = " ".join(map(str, answer_embedding.tolist()))
        else:
            answer_text = ""
            embedding_str = ""
        data.append({
            "ques_id": ques_id,
            "question": question,
            "answer": answer_text,
            "embedding": embedding_str
        })
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding="utf-8")

# 主流程
if __name__ == "__main__":
    # 加载问题文件
    questions_file_path = "./b榜/B_question.csv"
    questions_df = pd.read_csv(questions_file_path)

    # 第1步：文档解析
    folder_path = "./b榜/B_document"
    pdf_text = parse_pdfs_in_folder(folder_path)
    print("Parsed Text:", pdf_text[:500])

    # 第2步：加载模型
    model = SentenceTransformer("./model", device=device)

    # 第3步：文本分块
    max_chunk_length = 2048  # 定义最大字符长度
    chunks = split_text_into_chunks(pdf_text, max_chunk_length)
    print("Number of Chunks:", len(chunks))

    # 检查并截断超长的文本块
    for idx, chunk in enumerate(chunks):
        if len(chunk) > max_chunk_length:
            print(f"Chunk {idx} length {len(chunk)} exceeds max_chunk_length {max_chunk_length}. Truncating.")
            chunks[idx] = chunk[:max_chunk_length]

    # 计算最大字节长度
    max_chunk_length_bytes = max(len(chunk.encode('utf-8')) for chunk in chunks)
    print(f"Maximum chunk length in bytes after truncation: {max_chunk_length_bytes}")

    # 第4步：生成向量
    embeddings = generate_embeddings(chunks, model)

    # 第5步：向量存储入Milvus
    connections.connect("default", host="127.0.0.1", port="19530")

    collection_name = "knowledge_base"
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.drop()

    # 设置FieldSchema的max_length为最大字节长度的稍大值
    max_length = int(max_chunk_length_bytes * 1.1)  # 增加10%的余量
    if max_length > 65535:  # Milvus对max_length的上限
        max_length = 500
    print(f"Setting FieldSchema max_length to: {max_length}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embeddings.shape[1]),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=max_length)
    ]
    schema = CollectionSchema(fields, "Text embedding storage")
    collection = Collection(collection_name, schema)

    data_to_insert = [
        embeddings.tolist(),
        chunks
    ]

    # 插入数据时，指定字段名称（不包括'id'）
    mr = collection.insert(
        data=data_to_insert,
        fields=["embedding", "text"]
    )

    # **在这里添加创建索引的步骤**
    # 创建向量索引
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",  # 您也可以根据需要选择其他索引类型
        "params": {"nlist": 128},
    }
    print("Creating index...")
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    print("Index created.")

    # 加载集合
    collection.load()
    print(f"Inserted {len(chunks)} chunks into vector database.")

    # 第6步：文本召回
    results = []
    for q in questions_df["question"]:
        try:
            res = search_query(q, model, collection)
            results.append(res)
        except Exception as e:
            print(f"Error during search for question '{q}': {e}")
            results.append([])

    # 第7步：生成提交文件
    generate_submission_file(questions_df, results, model)
    print("Submission file generated: submit_example.csv")
