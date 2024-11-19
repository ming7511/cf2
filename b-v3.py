import os
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
import torch
import re
from multiprocessing import Pool, cpu_count
from functools import partial

# 确保 PyTorch 使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. 文档解析：解析PDF文档为文本数据
def parse_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            texts = [page.extract_text() for page in pdf.pages if page.extract_text()]
        return "\n".join(texts)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return ""

def parse_pdfs_in_folder(folder_path):
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    with Pool(processes=cpu_count()) as pool:
        texts = pool.map(parse_pdf, pdf_files)
    return "\n".join(texts)

# 2. 文本分割为句子（使用正则表达式）
def split_text_into_sentences(text):
    sentences = re.split(r'(?<=[。！？])\s*', text)
    return [s.strip() for s in sentences if s.strip()]

# 3. 向量化文本句子：使用 bge-large-zh-v1.5 模型
def generate_embeddings(text_list, model, batch_size=128):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            device=device,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# 4. 向量检索：基于Milvus的检索函数，使用余弦相似度
def search_query(question, model, collection, top_k=5):
    # 将问题向量化
    query_embedding = model.encode([question], device=device, convert_to_numpy=True)
    # 搜索向量数据库
    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )
    # 提取检索到的句子
    sentences = [result.entity.get("text") for result in results[0]]
    return sentences

# 5. 提交文件格式化，包含答案的嵌入
def generate_submission_file(questions_df, answers, model, output_file="submit_example.csv"):
    # 批量计算答案的嵌入向量
    answer_texts = [ans if ans else " " for ans in answers]
    embeddings = generate_embeddings(answer_texts, model)
    # 将嵌入向量转换为字符串格式，方便存储到CSV文件中
    embeddings_str = [" ".join(map(str, emb.tolist())) for emb in embeddings]
    # 创建提交文件
    submission_df = pd.DataFrame({
        "ques_id": questions_df["ques_id"],
        "question": questions_df["question"],
        "answer": answers,
        "embedding": embeddings_str
    })
    submission_df.to_csv(output_file, index=False, encoding="utf-8")

# 主流程
if __name__ == "__main__":
    # 加载问题文件
    questions_file_path = "./b榜/B_question.csv"
    questions_df = pd.read_csv(questions_file_path)

    # 第1步：文档解析
    folder_path = "./b榜/B_document"
    pdf_text = parse_pdfs_in_folder(folder_path)
    print("Parsed Text Sample:", pdf_text[:500])

    # 第2步：加载模型（使用 bge-large-zh-v1.5 模型）
    model_name = './model'  # 请确保已安装此模型
    model = SentenceTransformer(model_name, device=device)

    # 第3步：文本分割为句子
    sentences = split_text_into_sentences(pdf_text)
    print("Number of Sentences:", len(sentences))

    # 第4步：生成句子向量
    embeddings = generate_embeddings(sentences, model, batch_size=256)

    # 第5步：向量存储入Milvus
    connections.connect("default", host="127.0.0.1", port="19530")

    collection_name = "knowledge_base"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embeddings.shape[1]),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, "Sentence embedding storage")
    collection = Collection(collection_name, schema)

    # 插入数据
    data_to_insert = [embeddings.tolist(), sentences]
    collection.insert(
        data=data_to_insert,
        fields=["embedding", "text"]
    )

    # 创建向量索引（使用HNSW索引）
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 48, "efConstruction": 200},
    }
    print("Creating index...")
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    print("Index created.")

    # 加载集合
    collection.load()
    print(f"Inserted {len(sentences)} sentences into vector database.")

    # 第6步：文本召回并生成答案
    answers = []
    for q in questions_df["question"]:
        try:
            retrieved_sentences = search_query(q, model, collection)
            answer = ''.join(retrieved_sentences)
            # 限制答案长度
            max_answer_length = 400
            if len(answer) > max_answer_length:
                answer = answer[:max_answer_length] + '...'
            answers.append(answer)
        except Exception as e:
            print(f"Error during search for question '{q}': {e}")
            answers.append("")

    # 第7步：生成提交文件，包含答案的嵌入
    generate_submission_file(questions_df, answers, model)
    print("Submission file generated: submit_example.csv")
