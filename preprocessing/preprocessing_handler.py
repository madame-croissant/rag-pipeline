import os
import shutil

from preprocessing.chunking import ChunkStrategies
from preprocessing.vector_store import VectoreStore
from preprocessing.bm25 import BM25Indexer


class PreprocessingPipe:
    def __init__(self, db_path="./chroma_db", bm25_path="bm25_index.pkl"):
        self.db_path = db_path
        self.bm25_path = bm25_path


    def clear_database(self):
        """
        Clean up the db directory
        """
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

    def organiser(self, strategy_name: str, **kwargs) -> list[dict]:
        """
        1. claer database
        2. chunk docs using a chosen strategy
        3. store vectors in chromadb
        4. indexing chunks with bm25s
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

        #bm indexing
        bm25 = BM25Indexer()
        bm25.build_index(unique_chunks)
        bm25.save_index(self.bm25_path)

        print(f"\nPreprocessing pipeline completed. Indexed {len(unique_chunks)} unique chunks")
        return unique_chunks

