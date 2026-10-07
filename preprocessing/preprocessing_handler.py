"""
Preprocessing pipeline orchestration module.

Coordinates document chunking, vector database storage in ChromaDB,
and sparse BM25 indexing in a single unified workflow.
"""

import os
import shutil

from preprocessing.bm25 import BM25Indexer
from preprocessing.chunking import ChunkStrategies
from preprocessing.vector_store import VectoreStore


class PreprocessingPipe:
    """
    Orchestrates end-to-end preprocessing, chunking, and dual-indexing.

    Attributes:
        db_path (str): File system path for the persistent ChromaDB database.
        bm25_path (str): Directory or file path for persisting the BM25 index.
    """
    def __init__(self, db_path="./chroma_db", bm25_path="bm25_index.pkl"):
        """
        Initializes storage paths for vector and sparse indices.
        """
        self.db_path = db_path
        self.bm25_path = bm25_path


    def clear_database(self):
        """
        Removes existing ChromaDB database directory to ensure a fresh index.
        """
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

    def organiser(self, strategy_name: str, **kwargs) -> list[dict]:
        """
        Executes full preprocessing: 
            1. clean database
            2. chunk raw documents using a chosen strategy
            3. store dense vector embeddings in chromadb
            4. build the sparse BM25 index.

        Args:
            strategy_name (str): Chunking strategy to apply ('fixed', 'recursive', or 'semantic').
            **kwargs: Strategy-specific parameters (e.g., size, overlap, threshold).

        Returns:
            list[dict]: List of unique chunks that were embedded and indexed.

        Raises:
            ValueError: If an unsupported chunking strategy_name is provided.
        """

        self.clear_database()

        chunks = ChunkStrategies()

        print(f"Running chunking using strategy:{strategy_name}")

        if strategy_name == "fixed":
            strategy_func = chunks.fixed_size
            strategy_args = {
                "size": kwargs.get("size", 500),
                "overlap": kwargs.get("overlap", 50)
            }

        elif strategy_name == "recursive":
            strategy_func = chunks.recursive_split
            strategy_args = {
                "size": kwargs.get("size", 500),
                "separators": ["\n\n", "\n", ".", " "]
            }

        elif strategy_name == "semantic":
            strategy_func = chunks.semantic_split
            strategy_args = {
                "threshold": kwargs.get("threshold", 0.35),
                "min_chunk_size": kwargs.get("min_chunk_size", 300)
            }

        else:
            raise ValueError(f"Unknown strategy '{strategy_name}'. Valid options: ['fixed', 'recursive', 'semantic']")


        #Chunking
        raw_chunks = chunks.chunk_handler(
            strategy_func=strategy_func,
            strategy_name=strategy_name,
            **strategy_args
        )
   
        #Vector storing
        store = VectoreStore(db_path=self.db_path)
        unique_chunks = store.add_chunks(raw_chunks)

        #BM25 indexing
        bm25 = BM25Indexer()
        bm25.build_index(unique_chunks)
        bm25.save_index(self.bm25_path)

        print(f"\nPreprocessing pipeline completed. Indexed {len(unique_chunks)} unique chunks")
        return unique_chunks

