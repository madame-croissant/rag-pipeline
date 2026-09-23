import re
import json
from groq import Groq


class Evaluation:
    def __init__(self, key_path: str = "../keys.txt", model_name: str = "openai/gpt-oss-20b"):
        
        with open("../keys.txt", "r") as f:
            api_key = f.read().strip()
        
        self.api_key = api_key
        self.groq_client = Groq(api_key=api_key)
        self.model_name = model_name

    def judge_accuracy(self, query: str, golden_answer:str, candidate_answer: str) -> dict:
        """
        LLm compares cand answer to gold stand (1-5 score)
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
        
    def execute_evaluation(self, dataset_path = "eval_dataset.json"):

        from main import run_query
        
        with open(dataset_path, "r") as f:
            test_cases = json.load(f)
        
        results = []
        for case in test_cases:
            query = case.get("query")
            golden_answer = case.get("golden_answer")
            category = case.get("category", "direct_lookup")
            
            pipeline_output = run_query(query, key=self.api_key)

            ###checking the guardrail:

            if isinstance(pipeline_output, dict) and pipeline_output.get("intercepted"):
                fallback_msg = (
                    "I could not find sufficiently relevant information in the documents to"
                    " answer your request appropriately."
                )
                
                #extract the "message" from the nested "answer" dict:
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

            #####

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