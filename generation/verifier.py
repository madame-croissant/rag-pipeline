import json
import re
from groq import Groq


"""
Now all the text is here in verifyier -> needs to be moved into main later


"""

class CitationVerifier:
    def __init__(self, key: str, model_name: str = "openai/gpt-oss-20b"):
        self.client = Groq(api_key=key)
        self.model_name = model_name

    def verify_all(self, answer: str, chunks: list[dict]) -> dict:
        # 1. Map chunks by index 1, 2, 3...
        context_map = {idx: chunk.get("text", "") for idx, chunk in enumerate(chunks, 1)}

        # 2. Find all lines containing bracketed citations like [1], [2]
        lines = [line.strip() for line in answer.split("\n") if line.strip()]
        
        verifications = []
        supported_count = 0

        for line in lines:
            citations = re.findall(r'\[(\d+)\]', line)
            if not citations:
                continue

            # Clean line text for printing
            clean_claim = re.sub(r'^\d+\.\s*|^\*\s*|^-\s*', '', line)

            for cit_str in citations:
                c_idx = int(cit_str)
                context_text = context_map.get(c_idx, "")

                prompt = f"""Context:
{context_text}

Claim:
"{clean_claim}"

Does the Context explicitly support the Claim?
Answer ONLY with the word YES or NO."""

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )

                verdict = response.choices[0].message.content.strip().upper()
                is_supported = "YES" in verdict

                if is_supported:
                    supported_count += 1

                verifications.append({
                    "claim": clean_claim,
                    "citation_idx": c_idx,
                    "is_supported": is_supported
                })

        total = len(verifications)
        score = (supported_count / total * 100.0) if total > 0 else 0.0

        return {
            "verifications": verifications,
            "supported_count": supported_count,
            "total_citations": total,
            "verification_score": score
        }

if __name__ == "__main__":

    from retrieval.retrieval import HybridRetrieval
    from generation.generator import Generator
    from evaluation.scorer import ConfidenceScore
    from evaluation.guardrail import RetrievalGuardrail

    with open("../keys.txt", "r") as f:
        key = f.read().strip()


    query = "How are translations handled in FastAPI?"
    
    print("1. Retrieving context blocks")

    retriever = HybridRetrieval()
    reranked_chunks = retriever.hybrid_search(query=query)

    #TESTINNG FOR RETRIEVAL SCORE BEFORE GENERATION

    guardrail = RetrievalGuardrail(threshold=0.35)
    guard_result = guardrail.check(reranked_chunks, query)

    if not guard_result["passed"]:
        print("LOW RETRIEVAL SCORE - STOPPING BEFORE GENERATION")
        print(guard_result["fallback_response"])
        exit()

    
    
    ###### GENERATING  #######

    print("2. Generating answer with Groq")
    generator = Generator (key=key)
    verifier= CitationVerifier(key=key) 


    answer = generator.generate(query=query, chunks=reranked_chunks)
    
    print("\n Generated Answer")
    print(answer)

    print("Verifying Citations")

    verification_report = verifier.verify_all(answer, reranked_chunks)

    print("Verification Report")
    print(f"Total citations checked: {verification_report['total_citations']}")
    print(f"Supported citations:     {verification_report['supported_count']}")
    print(f"Verification Score:      {verification_report['verification_score']:.1f}%")


    for v in verification_report["verifications"]:
        status = "PASSED" if v["is_supported"] else "FAILED"
        c_idx = v.get("citation_idx", "?")
        claim_text = v.get("claim", "")

        print(f' - [{status}] Chunk [{c_idx}]: "{claim_text}"')


    #EVALUATION SCORES:

    print("\n Answer confidence score")

    scorer = ConfidenceScore()

    confidence_results = scorer.compute_scores(
        query=query,
        answer=answer,
        reranked_chunks=reranked_chunks,
        verification_report=verification_report
    )

    print("\nConfidence Report:")
    print(f"Composite Confidence Score: {confidence_results['composite_score'] * 100:.1f}%")
    print("Breakdown:")
    print(f"Retrieval Confidence: {confidence_results['breakdown']['retrieval_confidence'] * 100:.1f}%")
    print(f"Citation Coverage:    {confidence_results['breakdown']['citation_coverage'] * 100:.1f}%")
    print(f"Answer Completeness:  {confidence_results['breakdown']['answer_completeness'] * 100:.1f}%")

