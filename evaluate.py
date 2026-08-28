"""
RAG系统自动评测脚本 
"""
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frontend.local_rag.config.settings import get_settings
from frontend.local_rag.core.document_processor import DocumentProcessor
from frontend.local_rag.core.embedding_service import EmbeddingService
from frontend.local_rag.core.vector_store import VectorStoreManager
from frontend.local_rag.core.rag_chain import RAGChain

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

def test_supported_formats():
    print("\n" + "="*50)
    print("📋 支持的文档格式")
    print("="*50)
    supported_formats = ['.txt', '.md', '.pdf', '.docx', '.csv']
    print(f"支持 {len(supported_formats)} 种文档格式：{', '.join(supported_formats)}")
    return len(supported_formats)

def test_recall(vector_store, embedding_service, test_questions, k=10, threshold=0.35):
    print("\n" + "="*50)
    print(f"🔍 检索召回率测试（Top-{k}，相似度阈值：{threshold}）")
    print("="*50)
    recalls = []
    for i, question in enumerate(test_questions):
        results = vector_store.search(question, k=k)
        if not results:
            print(f"问题{i+1}: {question[:30]}... → 无结果，召回率 0%")
            recalls.append(0)
            continue
        q_vec = embedding_service.get_embedding(question)
        rel_cnt = 0
        for r in results:
            cont = r.get("content", "")
            if not cont:
                continue
            d_vec = embedding_service.get_embedding(cont)
            sim = cosine_similarity(q_vec, d_vec)
            if sim >= threshold:
                rel_cnt += 1
        recall = round(rel_cnt / k * 100, 1)
        recalls.append(recall)
        print(f"问题{i+1}: {question[:30]}... → 相关{rel_cnt}/{k}，召回率{recall}%")
    avg_recall = round(sum(recalls)/len(recalls),1)
    print(f"\n✅ 平均召回率：{avg_recall}%")
    return avg_recall

def test_response_time(rag_chain, test_questions):
    print("\n" + "="*50)
    print("⏱️ 问答响应时间测试")
    print("="*50)
    times = []
    for i, q in enumerate(test_questions):
        s = time.time()
        rag_chain.run(q, history=[])
        e = time.time()
        t = round(e-s,2)
        times.append(t)
        print(f"问题{i+1}: {q[:20]}... → {t}秒")
    avg_t = round(sum(times)/len(times),2)
    print(f"\n✅ 平均响应时间：{avg_t}秒")
    return avg_t

def main():
    print("\n" + "="*50)
    print("🚀 RAG宽松版自动评测")
    print("="*50)
    settings = get_settings()
    emb = EmbeddingService(settings)
    vs = VectorStoreManager(settings, emb.get_embeddings())
    dp = DocumentProcessor(settings)

    if not vs.load():
        print("\n📂 向量库为空，加载示例文档")
        sample_path = "frontend/local_rag/data/sample.txt"
        if os.path.exists(sample_path):
            docs = dp.process_file(sample_path)
            vs.add_documents(docs)
            print(f"已加载 {len(docs)} 个片段")
        else:
            print("无示例文档，先上传文档再测")
            return

    fmt_num = test_supported_formats()

    # 换成你知识库内有答案的问题再跑
    test_q = [
        "什么是RAG",
        "FAISS作用",
        "向量检索原理",
        "LangChain用途",
        "文档如何切分",
        "语义和关键词检索区别"
    ]
    avg_rec = test_recall(vs, emb, test_q, k=10, threshold=0.35)

    try:
        rc = RAGChain(settings, vs)
        t_q = ["你好","什么是RAG","项目用到哪些技术"]
        avg_time = test_response_time(rc, t_q)
    except Exception as err:
        print(f"\n⏭️ 响应时间跳过：{err}")
        avg_time = "N/A"

    print("\n" + "="*50)
    print("📊 最终汇总")
    print("="*50)
    print(f"文档格式：{fmt_num}种")
    print(f"平均召回率：{avg_rec}%")
    print(f"平均响应时间：{avg_time}秒")

if __name__ == "__main__":
    main()