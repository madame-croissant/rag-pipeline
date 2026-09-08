from retrieval.retrieval import HybridRetrieval
from groq import Groq


SYSTEM_PROMPT = """You are a precise technical documentation assistant.
Answer the user's question using ONLY the provided context blocks.

Formatting and Citation Rules:
1. Output your response as a numbered or bulleted list broken down by key steps or themes.
2. Every item or sentence MUST end with inline bracketed citations (e.g., [1], [2]) indicating the specific context block used.
3. Do NOT merge all citations into a single group at the end of a paragraph. Cite each specific point individually.
"""

class Generator:
    def __init__(self, key:str, model_name:str = "openai/gpt-oss-20b"):
        self.system_prompt = SYSTEM_PROMPT
        self.model_name = model_name
        self.client = Groq(api_key=key)

    def prep_context(self, chunks: list[dict]) -> str:
        """
        prep blocks to feed the  generator
        """
        formatted_blocks = []

        for idx, block in enumerate(chunks, 1):

            heading = block.get("metadata", {}).get("heading", "N/A") #if None -> N/A
            text = block.get("text")

            full_block = f"[{idx}] Heading: {heading} \nText:  {text}"

            formatted_blocks.append(full_block)

        return "\n\n".join(formatted_blocks)


    def generate(self, query:str, chunks:list[dict]) -> dict:
        """
        instructions + context + query -> prompt message
        """

        context_str = self.prep_context(chunks)

        user_message = f"Context blocks:\n{context_str} \n\nUser question: {query}"

        response = self.client.chat.completions.create(
            model = self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0
        )

        #Print exact token usage
        if response.usage:
            print(f"[Tokens Used] Prompt: {response.usage.prompt_tokens} | Completion: {response.usage.completion_tokens} | Total: {response.usage.total_tokens}")

        return response.choices[0].message.content


#Testing

if __name__ == "__main__":

    with open("../keys.txt", "r") as f:
        key= f.read().strip()

    

    retriever = HybridRetrieval()
    generator = Generator (key=key)

    query = "How are translations handled in FastAPI?"

    print("1. Retrieving context blocks")
    reranked_chunks = retriever.hybrid_search(query=query)

    print("2. Generating answer with Groq")
    answer = generator.generate(query=query, chunks=reranked_chunks)

    print("\n Answer")
    print(answer)


