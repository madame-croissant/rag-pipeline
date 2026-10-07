"""
BM25 sparse index management module.

Tokenizes document chunks using NLTK stemming and builds, saves, loads,
and searches sparse BM25 indices.
"""

import os
import json
import bm25s
from nltk.stem.snowball import SnowballStemmer

_snowball = SnowballStemmer(language="english")

def stem_word(words: str | list[str]) -> str | list[str]:
    """
    Stems a single word or list of words for the BM25 tokenizer.

    Args:
        words (str | list[str]): Word or list of words to stem.

    Returns:
        str | list[str]: Stemmed word or list of stemmed words.
    """

    if isinstance(words, str):
        return _snowball.stem(words)
    return [_snowball.stem(w) for w in words]

class BM25Indexer:
    """
    Manages sparse BM25 indexing and keyword search.

    Attributes:
        retriever (bm25s.BM25 | None): Active BM25 index instance.
        chunks (list[dict]): Document chunks backing the current index instance.
    """
    def __init__(self):
        """
        Initializes an empty BM25 indexer.
        """
        self.retriever = None
        self.chunks = []
        self.stemmer = stem_word

        def stemWord(self, word):
            return self.stemmer.stem(word)
    
    def build_index(self, chunks: list[dict]) -> None:
        """
        Tokenizes chunk texts and constructs the BM25 retrieval index.

        Args:
            chunks (list[dict]): List of chunk records containing 'text' keys.
        """
        self.chunks = chunks

        texts = [chunk["text"] for chunk in chunks]

        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, lower=True)

        #Indexes
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens)
    
    def save_index(self, save_dir: str = "bm25s_index") -> None:
        """
        Saves the BM25 index and corresponding chunk payload to disk.

        Args:
            save_dir (str): Directory path to persist index and chunk metadata.
        """
        
        os.makedirs(save_dir, exist_ok=True)
        self.retriever.save(save_dir)

        with open(os.path.join(save_dir, "bm25chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            #converting a python object into json

        print("BM25 indices and chunks saved")
    
    def load_index(self, save_dir: str = "bm25s_index") -> None:
        """
        Loads a pre-built BM25 index and chunk mapping from disk.

        Args:
            save_dir (str): Path to directory containing persisted index files.
        """
        
        self.retriever = bm25s.BM25.load(save_dir, load_corpus=False)

        with open(f"{save_dir}/bm25chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print("BM25 indices loaded")
    

    def search_sparse(self, query: str, k:int) -> list[dict]:
        """
        Executes keyword-based sparse search against indexed document chunks.

        Args:
            query (str): The search query prompt.
            k (int): Number of top matching chunks to retrieve.

        Returns:
            list[dict]: Retrieved documents formatted with text, metadata, and BM25 score.
        """

        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer, lower=True)

        results, scores = self.retriever.retrieve(query_tokens, k=k)

        sparse_results = []

        for doc_idx, score in zip(results[0], scores[0]):

            chunk = self.chunks[int(doc_idx)]
            sparse_results.append({
                "text": chunk["text"],
                "metadata": {
                    "heading": chunk.get("heading", "None"),
                    "source_file": chunk.get("source_file", "unknown")
                },
                "score": float(score),
                "source": "sparse"
            })
        
        return sparse_results
