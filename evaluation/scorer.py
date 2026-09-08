import re
import nltk
import math
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def compute_retrieval_score(reranked_chunks: list[dict]) -> float:
    """
    we use it twice:
    1. for retieval phase to remove the useless chunks before generation
    2. for confidence score after generation
    
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
    def __init__(self):
        self.stemmer = SnowballStemmer(language="english")
        #self.stop_words = set(stopwords.words("english"))

        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            self.stop_words = set(stopwords.words("english"))

    def compute_completeness(self, query: str, answer: str) -> float:
        """
       how many stemmed query tokens appear in the generated answer text 
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
            query,
            answer,
            reranked_chunks: list[dict],
            verification_report: dict,
            weights=(0.40, 0.40, 0.20)) -> dict: #weights are importance of each score 40 %, 40%, 20%

        #what weights mean

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