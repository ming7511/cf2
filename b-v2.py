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

# 1. 文档解析：解析PDF文档为文本数据
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

# 2. 文本分割为句子
def split_text_into_sentences(text):
    # 使用正则表达式按照句子结束符进行分割
    sentences = re.split('(?<=[。！？])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# 3. 向量化文本句子：使用 bge-large-zh-v1.5 模型
def generate_embeddings(text_list, model):
    embeddings = model.encode(
        text_list,
        device=device,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=128  # 根据显存大小调整
    )
    return embeddings

# 4. 向量检索：基于Milvus的检索函数，使用余弦相似度
def search_query(question, model, collection, top_k=5):
    # 将问题向量化
    query_embedding = model.encode([question], device=device, convert_to_numpy=True)

    # 搜索向量数据库
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k * 2,  # 检索更多的句子以供筛选
        output_fields=["text"]
    )

    # 提取检索到的句子
    sentences = [result.entity.get("text") for result in results[0]]
    scores = [result.score for result in results[0]]

    # 关键词匹配，计算与问题的词汇重合度
    question_words = set(question)
    sentence_scores = []
    for sentence, score in zip(sentences, scores):
        sentence_words = set(sentence)
        overlap = len(question_words & sentence_words)
        sentence_scores.append((sentence, score, overlap))

    # 根据重合度和向量相似度进行排序
    sentence_scores.sort(key=lambda x: (x[2], x[1]), reverse=True)

    # 选取前几个句子作为答案
    selected_sentences = [s[0] for s in sentence_scores[:top_k]]

    # 拼接答案，限制答案长度
    max_answer_length = 400  # 可以根据需要调整
    answer = ''.join(selected_sentences)
    if len(answer) > max_answer_length:
        answer = answer[:max_answer_length] + '...'

    return answer

# 5. 提交文件格式化，包含答案的嵌入
def generate_submission_file(questions, results, model, output_file="submit_example.csv"):
    data = []
    for ques_id, question, answer in zip(questions["ques_id"], questions["question"], results):
        if answer:
            # 计算答案的嵌入向量
            answer_embedding = model.encode([answer], device=device, convert_to_numpy=True)[0]
            # 将嵌入向量转换为字符串格式，方便存储到CSV文件中
            embedding_str = " ".join(map(str, answer_embedding.tolist()))
        else:
            embedding_str = ""
        data.append({
            "ques_id": ques_id,
            "question": question,
            "answer": answer,
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
    print("Parsed Text Sample:", pdf_text[:500])

    # 第2步：加载模型（使用 bge-large-zh-v1.5 模型）
    model_name = './model'  # 请确保已安装此模型
    model = SentenceTransformer(model_name, device=device)

    # 第3步：文本分割为句子
    sentences = split_text_into_sentences(pdf_text)
    print("Number of Sentences:", len(sentences))

    # 第4步：生成句子向量
    embeddings = generate_embeddings(sentences, model)

    # 第5步：向量存储入Milvus
    connections.connect("default", host="127.0.0.1", port="19530")

    collection_name = "knowledge_base"
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.drop()

    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embeddings.shape[1]),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, "Sentence embedding storage")
    collection = Collection(collection_name, schema)

    # 插入数据
    data_to_insert = [
        embeddings.tolist(),
        sentences
    ]
    mr = collection.insert(
        data=data_to_insert,
        fields=["embedding", "text"]
    )

    # 创建向量索引（使用HNSW索引）
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 100},
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
    results = []
    for q in questions_df["question"]:
        try:
            answer = search_query(q, model, collection)
            results.append(answer)
        except Exception as e:
            print(f"Error during search for question '{q}': {e}")
            results.append("")

    # 第7步：生成提交文件，包含答案的嵌入
    generate_submission_file(questions_df, results, model)
    print("Submission file generated: submit_example.csv")
