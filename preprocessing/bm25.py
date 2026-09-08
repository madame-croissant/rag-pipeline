import bm25s
import json
import os
from nltk.stem.snowball import SnowballStemmer



_snowball = SnowballStemmer(language="english")

def stem_word(words):
    """
    bm25s passes a list of words or a single word string. Needs a single word string
    """
    if isinstance(words, str):
        return _snowball.stem(words)
    return [_snowball.stem(w) for w in words]

class BM25Indexer:
    def __init__(self):
        self.retriever = None
        self.chunks = []
        self.stemmer = stem_word

        def stemWord(self, word):
            return self.stemmer.stem(word)
    
    def build_index(self, chunks: list[dict]):
        self.chunks = chunks

        texts = [chunk["text"] for chunk in chunks]

        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, lower=True)

        #Indexes
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens)
    
    def save_index(self, save_dir="bm25s_index"):

        os.makedirs(save_dir, exist_ok=True)

        self.retriever.save(save_dir)

        with open(os.path.join(save_dir, "bm25chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            #converting a python object into json

        print("bm25s index and chunks saved")
    
    def load_index(self, save_dir="bm25s_index"):
        
        self.retriever = bm25s.BM25.load(save_dir, load_corpus=False)

        with open(f"{save_dir}/bm25chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print("bm25s index loaded")
    

    def search_sparse(self, query: str, k:int) -> list:

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
