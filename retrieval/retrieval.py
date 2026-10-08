"""
Hybrid retrieval and re-ranking module.

Combines dense vector search (ChromaDB) and sparse keyword search (BM25) using 
Reciprocal Rank Fusion (RRF), followed by Cross-Encoder re-ranking.
"""

from sentence_transformers import CrossEncoder

from preprocessing.bm25 import BM25Indexer
from preprocessing.vector_store import VectoreStore




class HybridRetrieval:
    """
    Orchestrates hybrid search using dense vector, sparse keyword, RRF fusion, and cross-encoder re-ranking.

    Attributes:
        vectore_store (VectoreStore): Persistent ChromaDB dense search engine.
        bm25 (BM25Indexer): Sparse BM25 keyword search engine.
        reranker (CrossEncoder): Hugging Face Cross-Encoder model used to re-rank candidate chunks.
    """

    def __init__(self, db_path: str = "./chroma_db", bm25_path: str = "./bm25_index.pkl") -> None:
        """
        Loads dense and sparse indices along with the re-ranking model.

        Args:
            db_path (str): File system path to ChromaDB database.
            bm25_path (str): Directory or file path containing saved BM25 index.
        """
        
        self.vectore_store = VectoreStore(db_path=db_path)
        self.bm25 = BM25Indexer()
        self.bm25.load_index(save_dir=bm25_path)

        #cross-encoder for reranking
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def get_dense(self, query: str, k: int = 10) -> list[dict]:
        """
        Queries the dense vector store for semantic matches.

        Args:
            query (str): The search query.
            k (int): Number of top results to retrieve.

        Returns:
            list[dict]: List of dense retrieval candidate dicts.
        """
        return self.vectore_store.search_dense(query=query, k=k) 
    
    def get_sparse(self, query: str, k: int = 10) -> list[dict]:
        """
        Queries the BM25 index for keyword matches.

        Args:
            query (str): The search query.
            k (int): Number of top results to retrieve.

        Returns:
            list[dict]: List of sparse retrieval candidate dicts.
        """
        return self.bm25.search_sparse(query=query, k=k)

    def rerank(self, query: str, candidates: list[dict], top_k=5) -> list[dict]:
        """
        Re-ranks candidate document chunks using the Cross-Encoder model.

        Args:
            query (str): The target user query.
            candidates (list[dict]): Combined candidate chunks from RRF fusion.
            top_k (int): Number of top candidates to return after re-ranking.

        Returns:
            list[dict]: Top candidate chunks sorted by descending cross-encoder score.
        """
        pairs = []

        for entry in candidates: #{"text": "FastAPI uses Pydantic for data validation...", "score": 0.016}
            pairs.append([query, entry["text"]])

        rerank_scores = self.reranker.predict(pairs)

        #Adding rerank scores
        for entry, score in zip(candidates, rerank_scores):
            entry["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]



    def hybrid_search(self, query: str, top_k: int = 10, rrf_k: int = 60, alpha: float = 0.5) -> list[dict]:
        """
        Executes full hybrid search: dense + sparse retrieval, RRF fusion, and Cross-Encoder re-ranking.

        Applies weighted Reciprocal Rank Fusion:
            score = weight * (1 / (rrf_k + idx))

        Args:
            query (str): Search prompt or question.
            top_k (int): Number of candidates to retrieve per branch and return after re-ranking.
            rrf_k (int): Reciprocal Rank Fusion smoothing parameter (default 60).
            alpha (float): Weight factor balancing dense (alpha) vs sparse (1 - alpha) results.

        Returns:
            list[dict]: Final re-ranked list of top document chunks.
        
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

