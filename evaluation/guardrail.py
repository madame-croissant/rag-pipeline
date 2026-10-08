"""
Retrieval confidence guardrail module.

Evaluates the relevance score of retrieved context chunks against a defined threshold
to prevent hallucination by intercepting low-confidence queries before generation.
"""

from evaluation.scorer import compute_retrieval_score

class RetrievalGuardrail:
    """
    Evaluates chunk relevance scores to intercept low-confidence retrieval attempts.

    1. If the top retrieval score (best chunk) falls below threshold -> blocks generation and returns fallback response.
    2. If retrieval score passes threshold (some bad chunks) -> permits generation with retrieved context.

    Attributes:
        threshold (float): Minimum acceptable retrieval confidence score (default 0.35).
    """

    def __init__(self, threshold: float = 0.35):
        """
        Initializes guardrail with a minimum retrieval confidence threshold.

        Args:
            threshold (float): Cutoff score below which queries are intercepted.
        """
        self.threshold = threshold


    def check(self, reranked_chunks: list[dict], query: str = "") -> dict:
        """
        Checks retrieved chunks against the threshold score.

        If retrieval score is too low, extracts top candidate headings as alternative 
        suggestions and builds a fallback response.

        Args:
            reranked_chunks (list[dict]): Re-ranked candidate chunks with scores and metadata.
            query (str): The original search query string.

        Returns:
            dict: Guardrail check result containing 'passed' status, 'confidence' score, 
                  and optional 'fallback_response'.
        """

        retrieval_score = compute_retrieval_score(reranked_chunks)
        rounded_score = round(retrieval_score, 3)

        if retrieval_score < self.threshold:

            suggested_sections = set()

            for chunk in reranked_chunks:
                metadata = chunk.get("metadata", {})
                heading = metadata.get("heading", "Untitled Section")
                suggested_sections.add(heading)

            suggested_sections = list(suggested_sections)
            top_three_sections = suggested_sections[:3]
         

            fallback_response = {}
            fallback_response["status"] = "low_confidence"
            fallback_response["query"] = query
            fallback_response["retrieval_confidence"] = rounded_score
            fallback_response["message"] = (
                "I could not find sufficiently relevant information in the documents to answer your request appropriately"
            )
            fallback_response["suggested_sections"] = top_three_sections

            #in case of fail
            result = {}
            result["passed"] = False
            result["confidence"] = rounded_score
            result["fallback_response"] = fallback_response
            return result

        # fo sucess
        result = {}
        result["passed"] = True
        result["confidence"] = rounded_score
        result["fallback_response"] = None
        return result