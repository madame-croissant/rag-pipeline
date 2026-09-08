from preprocessing.vector_store import VectoreStore
from preprocessing.bm25 import BM25Indexer

from sentence_transformers import CrossEncoder


class HybridRetrieval:
    def __init__(self, db_path="./chroma_db", bm25_path="./bm25_index.pkl"):
        
        self.vectore_store = VectoreStore(db_path=db_path)

        self.bm25 = BM25Indexer()
        self.bm25.load_index(save_dir=bm25_path)

        #cross-encoder for reranking
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def get_dense(self, query, k=10):
        return self.vectore_store.search_dense(query=query, k=k) 
    
    def get_sparse(self, query, k=10):
        return self.bm25.search_sparse(query=query, k=k)

    def rerank(self, query: str, candidates: list[dict], top_k=5) -> list[dict]:

        pairs = []

        for entry in candidates: #{"text": "FastAPI uses Pydantic for data validation...", "score": 0.016}
            pairs.append([query, entry["text"]])

        rerank_scores = self.reranker.predict(pairs)

        #adding rerank scores
        for entry, score in zip(candidates, rerank_scores):
            entry["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]



    def hybrid_search(self, query, top_k=10, rrf_k=60, alpha=0.5):
        """
        rrf_score = 1 / (60 + ranking_index)
        """

        dense_results = self.get_dense(query, k=top_k)
        sparse_results = self.get_sparse(query, k=top_k)

        fused_scores = {}

        for idx, res in enumerate(dense_results, 1):
            text = res["text"]
            rrf_score = alpha * (1 /  (rrf_k + idx))

            fused_scores[text] = {
                "text": text,
                "metadata": res["metadata"],
                "score": rrf_score
                }

        for idx, res in enumerate(sparse_results, 1):
            text = res["text"]
            rrf_score = (1 - alpha) * (1 /  (rrf_k + idx))

            if text in fused_scores:
                fused_scores[text]["score"] += rrf_score #if the same text, we add rrf score
            else:
                fused_scores[text] = {
                                "text": text,
                                "metadata": res["metadata"],
                                "score": rrf_score
                                }

        #sort the results

        fused_results = sorted(
            fused_scores.values(), key=lambda item: item["score"], reverse=True
        )

        reranked_results = self.rerank(query, fused_results, top_k)

        return reranked_results


        

    

#Testing
if __name__ == "__main__":
    retriever = HybridRetrieval()

    query_txt = "How are translations handled in FastAPI?"

    print("1.Retrieval")

    reranked_results = retriever.hybrid_search(query=query_txt)

    for idx, hit in enumerate(reranked_results, 1):
            print(f"\n {idx}, score {hit["score"]}, heading {hit["metadata"].get("heading")}")
            print(f"Text: {hit["text"]}")

    #dense_results = retriever.get_dense(query_txt, k=5)


    #print(f"Retrieved {len(dense_results)} chunks")
    #for idx, hit in enumerate(dense_results, 1):
    #    print(f"\n {idx}, score {hit["score"]}, heading {hit["metadata"].get("heading")}")
    #   print(f"Text: {hit["text"]}")
    
    #print("2. Sparse Retrieval")

    #sparse_results = retriever.get_sparse(query_txt, k=5)
    #for idx, hit in enumerate(sparse_results, 1):
    #    print(f"\n {idx}, score {hit["score"]}, heading {hit["metadata"].get("heading")}")
    #    print(f"Text: {hit["text"]}")
