"""
Evaluation module.

Uses an LLM judge via Groq to evaluate generated RAG candidate responses 
against ground-truth golden answers on a 1–5 scale, handling out-of-domain 
guardrail interceptions.
"""


import re
import json
from groq import Groq


class Evaluation:
    """
    Manages automated evaluation of RAG responses against benchmark test cases.

    Attributes:
        api_key (str): Groq API key loaded from file.
        groq_client (Groq): Initialized Groq API client instance.
        model_name (str): LLM judge model identifier.
    """
    
    def __init__(self, key_path: str = "../keys.txt", model_name: str = "openai/gpt-oss-20b") -> None:
        """
        Initializes the Groq client with an API key read from disk.

        Args:
            key_path (str): File system path to the plain-text API key file.
            model_name (str): Identifier of the LLM model to use as evaluator.
        """
        
        with open(key_path, "r") as f:
            api_key = f.read().strip()
        
        self.api_key = api_key
        self.groq_client = Groq(api_key=api_key)
        self.model_name = model_name

    def judge_accuracy(self, query: str, golden_answer: str, candidate_answer: str) -> dict:
        """
        Evaluates candidate response factual accuracy against a golden ground-truth answer.

        Prompts the judge LLM to assign a score from 1 to 5 along with a brief explanation.

        Args:
            query (str): The original user query.
            golden_answer (str): Ground-truth expected answer.
            candidate_answer (str): Response text produced by the RAG pipeline.

        Returns:
            dict[str, int | str]: Dictionary containing 'score' (1-5) and 'reason' string.

        """

        prompt = f"""You are an objective AI evaluator.
        Compare the candidate answer against the golden fround-truth answer for the given question.

        Question: {query}
        Golden Answer: {golden_answer}
        Candidate Answer: {candidate_answer}

        Rate the candidate answer on a scale of 1 to 5:
        1 - Completely wrong ot missing key information.
        2 - Partially correct but missing major details.
        3 - Mostly accurate with minor omissions.
        4 - Accurate and covers all primary facts.
        5 - Perfect factual match to the golden answer.

        Format your response EXACTLY as follows (two lines only):
        SCORE: <integer 1-5>
        REASON: <one sentence explanation>"""

        try:
            response = self.groq_client.chat.completions.create(
                model= self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            content = response.choices[0].message.content.strip()
            

            score_match = re.search(r"SCORE: \s*([1-5])", content)
            score = int(score_match.group(1)) if score_match else 1

            reason_match = re.search(r"REASON: \s*(.*)", content)
            reason = reason_match.group(1).strip() if reason_match else content

            return {"score": score, "reason": reason}
        
        except Exception as e:
            return {"score": 1, "reason": f"Evaluation error: {str(e)}"}
        
    def execute_evaluation(self, dataset_path: str = "eval_dataset.json") -> list[dict]:
        """
        Executes full evaluation suite over a JSON dataset of query/golden-answer test cases.

        Evaluates answer accuracy and verifies guardrail interception for out-of-domain queries.

        Args:
            dataset_path (str): File path to evaluation test dataset JSON.

        Returns:
            list[dict]: Detailed benchmark results containing candidate answers, scores, and reasons.
        """

        from main import run_query
        
        with open(dataset_path, "r") as f:
            test_cases = json.load(f)
        
        results = []
        for case in test_cases:
            query = case.get("query")
            golden_answer = case.get("golden_answer")
            category = case.get("category", "direct_lookup")
            
            pipeline_output = run_query(query, key=self.api_key)

            #Checking the guardrail:

            if isinstance(pipeline_output, dict) and pipeline_output.get("intercepted"):
                fallback_msg = (
                    "I could not find sufficiently relevant information in the documents to"
                    " answer your request appropriately."
                )
                
                #Extract the "message" from the nested "answer" dict:
                answer = pipeline_output.get("answer", {})
                candidate_answer = answer.get("message", fallback_msg)
                

                if category == "out_of_domain":
                    score = 5
                    reason = "Corrrectly intercepted out_of_domain query via Guardrail."
                else:
                    score = 1
                    reason = "Guardrail incorrectly blocked a valid query"
            else:
                if isinstance(pipeline_output, dict):
                    candidate_answer = pipeline_output.get("answer", "")
                else:
                    candidate_answer = str(pipeline_output)
                    

                eval_res = self.judge_accuracy(query=query, golden_answer=golden_answer, candidate_answer=candidate_answer)

                score = eval_res["score"]
                reason = eval_res["reason"]

            results.append({
                "query": query,
                "golden_answer": golden_answer,
                "candidate_answer": candidate_answer,
                "score": score,
                "reason": reason,
            })
        return results