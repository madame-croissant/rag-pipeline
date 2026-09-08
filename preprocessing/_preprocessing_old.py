from preprocessing.chunking import ChunkStrategies
from preprocessing.vector_store import VectoreStore
from preprocessing.bm25 import BM25Indexer

import os
import shutil


if __name__ == "__main__":

    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    chunks = ChunkStrategies()
   
    semantic_chunks = chunks.chunk_handler(
        chunks.semantic_split, 
        "semantic", 
        threshold=0.35, 
        min_chunk_size=300
        )
    
    store = VectoreStore()
    unique_chunks = store.add_chunks(semantic_chunks)

    #results = store.collection.query(
    #    query_texts=["FastAPI translations"],
    #    n_results=2
    #)

    #print("\n Test Retrieval Resuls")
    #print("Found text:", results["documents"][0][0])

    bm25 = BM25Indexer()
    bm25.build_index(unique_chunks)
    bm25.save_index("bm25_index.pkl")

