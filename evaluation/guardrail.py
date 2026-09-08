from evaluation.scorer import compute_retrieval_score

class RetrievalGuardrail:
    """
    checking the score of retrived chunks:
    1. if best chunk is low -> stopping
    2. if some bad chunks -> taking and proceeding to generation
    """

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold


    def check(self, reranked_chunks: list[dict], query: str = "") -> dict:
        """
        if retrival score is too low -> stopping
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