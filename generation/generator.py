"""
RAG response generation module.

Formats retrieved context chunks into indexed blocks and generates grounded 
answers with inline bracketed citations using the Groq API.
"""

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
    """
    Manages context block formatting and LLM response generation via Groq.

    Attributes:
        system_prompt (str): Core instructions enforcing strict grounded citations.
        model_name (str): Groq LLM model identifier.
        client (Groq): Initialized Groq API client instance.
    """
    def __init__(self, key:str, model_name:str = "openai/gpt-oss-20b"):
        """
        Initializes the Generator with API credentials and model configuration.

        Args:
            key (str): Plain-text Groq API key string.
            model_name (str): LLM model identifier to execute generation requests.
        """
        self.system_prompt = SYSTEM_PROMPT
        self.model_name = model_name
        self.client = Groq(api_key=key)

    def prep_context(self, chunks: list[dict]) -> str:
        """
        Formats context chunks into numbered blocks with headings for the generator.

        Args:
            chunks (list[dict]): Retrieved context chunks with metadata and text.

        Returns:
            str: Single formatted string containing all numbered context blocks.
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
        Constructs prompt, requests LLM completion, and logs API token usage.

        Args:
            query (str): The user's prompt or technical question.
            chunks (list[dict]): Re-ranked candidate context chunks.

        Returns:
            str: Generated completion text with inline citations.
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

