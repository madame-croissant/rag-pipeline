"""
Scoring and confidence evaluation module.

Computes normalized retrieval scores using sigmoid logit transformation and calculates
a composite confidence score based on retrieval confidence, citation accuracy, and 
query completeness.
"""

import re
import nltk
import math
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def compute_retrieval_score(reranked_chunks: list[dict]) -> float:
    """
    Computes a normalized retrieval score from candidate chunks. Applied twice:
    1. for retieval phase to remove the low-scored chunks before generation
    2. for confidence score after generation

    Applies a sigmoid transformation to Cross-Encoder logits (0.0 to 1.0 probability)
    or scales raw RRF scores against maximum theoretical RRF value (~0.033).

    Args:
        reranked_chunks (list[dict]): List of candidate chunks with scores.

    Returns:
        float: Normalized retrieval confidence score between 0.0 and 1.0.
    
    """

    if reranked_chunks and "rerank_score" in reranked_chunks[0]:
        raw_logit = float(reranked_chunks[0]["rerank_score"])

        # Sigmoid transform converts Cross-Encoder logits into 0.0 - 1.0 probability (normalisation)

        retrieval_score = 1.0 / (1.0 + math.exp(-raw_logit))
        

    elif reranked_chunks:
        # if reranked not available , use RRF score scaled against max RRF (~0.033)

        raw_rrf = float(reranked_chunks[0].get("score", 0.0))
        retrieval_score = min(max(raw_rrf / 0.033, 0.0), 1.0)

    else:
        retrieval_score = 0.0

    return retrieval_score

    

class ConfidenceScore:
    """
    Computes composite answer confidence scores combining retrieval, citation, and query metrics.

    Attributes:
        stemmer (SnowballStemmer): English language stemmer for keyword normalization.
        stop_words (set[str]): Set of English stop words to filter out during completeness scoring.
    """
    
    def __init__(self):
        """
        Initializes the stemmer and ensures English NLTK stopwords are downloaded.
        """
        
        self.stemmer = SnowballStemmer(language="english")

        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            self.stop_words = set(stopwords.words("english"))

    def compute_completeness(self, query: str, answer: str) -> float:
        """
        Calculates how many how many stemmed query tokens appear in the generated answer text. 

        Args:
            query (str): The user search prompt or question.
            answer (str): The generated RAG response text.

        Returns:
            float: Completeness ratio between 0.0 and 1.0.
       
        """

        query_words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        keywords = [
            self.stemmer.stem(w) 
            for w in query_words 
            if w not in self.stop_words
        ]
        answer_text = answer.lower()

        matched = sum(1 for kw in keywords if kw in answer_text)

        return matched / len(keywords)

    def compute_scores(
            self,
            query: str,
            answer: str,
            reranked_chunks: list[dict],
            verification_report: dict,
            weights=(0.40, 0.40, 0.20)) -> dict: #weights are importance of each score 40 %, 40%, 20%
        
        """Calculates a weighted composite confidence score and breakdown.

        Combines:
            - Retrieval confidence (40% weight by default)
            - Citation coverage ratio (40% weight by default)
            - Answer completeness ratio (20% weight by default)

        Args:
            query (str): The user query string.
            answer (str): The generated response text.
            reranked_chunks (list[dict]): Retrieved context chunks.
            verification_report (dict): Verification results with total and supported citation counts.
            weights (tuple[float, float, float]): Weights for (retrieval, citation, completeness).

        Returns:
            dict: Dictionary containing 'composite_score' and sub-score 'breakdown'.
        """

        #What weights mean

        w_retrieval = weights[0]
        w_citation = weights[1]
        w_completeness = weights[2]

        #Retrieval score

        retrieval_score = compute_retrieval_score(reranked_chunks)

        #Citation score
        total_cits = verification_report.get("total_citations", 0)
        supported_cits = verification_report.get("supported_count", 0)
        
        if total_cits > 0:
            citation_score = supported_cits / total_cits
        else:
            citation_score = 1.0
        
        #Completeness score
        completeness_score = self.compute_completeness(query, answer)

        composite = (
            (w_retrieval * retrieval_score) +
            (w_citation * citation_score) +
            (w_completeness * completeness_score)
        )

        return {
            "composite_score": round(composite, 3),
            "breakdown":{
                "retrieval_confidence": round(retrieval_score, 3),
                "citation_coverage": round(citation_score, 3),
                "answer_completeness": round(completeness_score, 3)
            }
        }