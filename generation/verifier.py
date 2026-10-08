
"""
Citation verification module.

Parses bracketed citations ([1], [2]) from generated RAG answer lines and uses an LLM 
verifier via Groq to confirm whether cited context blocks explicitly support each claim.
"""

import re
from groq import Groq


class CitationVerifier:
    """
    Verifies factual alignment between generated claim statements and their cited context chunks.

    Attributes:
        client (Groq): Initialized Groq API client instance.
        model_name (str): Groq LLM model identifier used for verification judgements.
    """
    
    def __init__(self, key: str, model_name: str = "openai/gpt-oss-20b") -> None:
        """
        Initializes the verifier with API credentials and model configuration.

        Args:
            key (str): Plain-text Groq API key string.
            model_name (str): LLM model identifier to execute verification requests.
        """
        self.client = Groq(api_key=key)
        self.model_name = model_name

    def verify_all(self, answer: str, chunks: list[dict]) -> dict:
        """
        Extracts claims and bracketed citations from an answer string and verifies each against its chunk.

        Args:
            answer (str): Generated answer text containing inline bracketed citations (e.g., [1], [2]).
            chunks (list[dict]): List of retrieved context chunks used to construct the answer.

        Returns:
            dict: Verification report containing individual claim results, supported count, 
                  total citations checked, and overall percentage score.
        """
        # Map chunks by index 1, 2, 3...
        context_map = {idx: chunk.get("text", "") for idx, chunk in enumerate(chunks, 1)}

        # Find all lines containing bracketed citations like [1], [2]
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
