"""
Document chunking strategies module.

Applies various text segmentation algorithms: fixed-size window, recursive hierarchical 
splitting, or semantic embedding distance—to structured sections.
"""


import re
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from preprocessing.loader import DataLoader


class ChunkStrategies:
    """
    Manages document chunking using fixed, recursive, and semantic splitting algorithms.

    Attributes:
        loader (DataLoader): DataLoader instance containing structured sections.
        encoder (SentenceTransformer): Embedding model used to compute sentence-level 
            similarity for semantic splitting.
    """
    def __init__(self) -> None:
        """
        Initializes the DataLoader and loads the sentence embedding model.
        """
        self.loader = DataLoader()
        self.loader.build_corpus('data/raw')

        #model for semantic split
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")


    def chunk_handler(self, strategy_func: str, strategy_name: str, **kwargs) -> list[dict]:
        """
        Applies a selected chunking strategy across all loaded document sections.

        Args:
            strategy_func: The chunking function to execute (e.g., `fixed_size`, `recursive_split`, or `semantic_split`).
            strategy_name (str): Identifier label for the chunking method used.
            **kwargs: Strategy-specific parameters (e.g., size, overlap, threshold).

        Returns:
            list[dict[str, str]]: List of final chunk records containing text and metadata.
        """
        final_chunked_text = []

        print("Total entries to chunk:", len(self.loader.dict_list))

        for dict in tqdm(self.loader.dict_list, desc="Chunking"):
            
            #pieces = self.fixed_size(dict["text"], size=500, overlap=50)
            pieces = strategy_func(dict["text"], **kwargs)

            for piece in pieces:
                new_chunk = {
                    "source_file": dict["source_file"],
                    "heading": dict["heading"],
                    "text": piece,
                    "chunking_strategy": strategy_name,
                }

                final_chunked_text.append(new_chunk)
        
        print("Total chunks after splitting:", len(final_chunked_text))

        return final_chunked_text

    def fixed_size(self, text: str, size: int, overlap: int) -> list[str]:
        """
        Splits text into fixed-length character windows with overlapping characters.

        Args:
            text (str): Input text string.
            size (int): Target chunk size in characters.
            overlap (int): Number of overlapping characters between consecutive chunks.

        Returns:
            list[str]: Extracted text chunks.
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + size
            piece = text[start : end]

            chunks.append(piece)

            start = start + (size - overlap)

        return chunks
    
    def recursive_split(self, text: str, size: int, separators: list) -> list[str]:
        """
        Recursively splits text using a hierarchy of separators until blocks reach `size`.

        Tries splitting on high-level separators first, falling back to smaller separators 
        only when chunks exceed the target size.

        Args:
            text (str): Input text string.
            size (int): Maximum target chunk size in characters.
            separators (list[str]): Ordered list of string delimiters to split on.

        Returns:
            list[str]: List of recursively bounded text chunks.
        """

        if len(text) <= size:
            return [text] #should be a list

        if not separators: #empty list
            return self.fixed_size(text, size, overlap=50)

        current_separator = separators[0]
        remaining_separators = separators[1:]

        #split the text on a current separator
        raw_pieces = text.split(current_separator)

        chunks = []
        current_chunk = ""

        #if small pieces -> merge together to a size
        for piece in raw_pieces:
            if not piece.strip():
                continue
            
            if len(piece) > size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                sub_chunks = self.recursive_split(piece, size, remaining_separators)
                chunks.extend(sub_chunks)
            
            elif len(current_chunk) + len(piece) +len(current_separator) <= size:
                if current_chunk:
                    current_chunk += current_separator + piece
                else:
                    current_chunk = piece
            else:
                chunks.append(current_chunk.strip())
                current_chunk = piece
        if current_chunk:
            chunks.append(current_chunk.strip())
      
        return chunks
    
    def semantic_split(self, text: str, threshold: float, min_chunk_size: int) -> list[str]:
        """
        Splits text into semantic chunks based on cosine similarity of consecutive sentences.

        Embeds sentences using `SentenceTransformer` and calculates cosine similarity between 
        adjacent sentences. Creates a chunk boundary when similarity drops below `threshold` 
        and the accumulated text meets `min_chunk_size`.

        Args:
            text (str): Input text string.
            threshold (float): Minimum cosine similarity score required to keep sentences grouped.
            min_chunk_size (int): Minimum character length required before allowing a split.

        Returns:
            list[str]: Semantically coherent text chunks.
        """
        
        sentences = self.split_into_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []
        
        #sent -> vect

        embeddings = self.encoder.encode(sentences)

        #similarity
        similarities = []
        for i in range(len(embeddings) - 1):

            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)

        chunks = []
        current_chunk_sent = [sentences[0]]

        for i in range(len(similarities)):
            current_sim = similarities[i]
            next_sentence = sentences[i + 1]

            current_text_len = len(" ".join(current_chunk_sent))

            if current_sim < threshold and current_text_len >= min_chunk_size:
                #topic shift -> current gr a single chunk
                chunks.append(" ".join(current_chunk_sent))

                #new group
                current_chunk_sent = [next_sentence]
            else:
                #same topic -> cont building
                current_chunk_sent.append(next_sentence)
        #save final group
        if current_chunk_sent:
            chunks.append(" ".join(current_chunk_sent))
        
        return chunks

    def split_into_sentences(self, text: str) -> list[str]:
        """
        Splits raw text into individual sentences using punctuation delimiters.

        Args:
            text (str): Input text string.

        Returns:
            list[str]: List of cleaned sentence strings.
        """

        raw_sent = re.split(r'(?<=[.!?])\s+', text)

        sentences = [s.strip() for s in raw_sent if s.strip()]

        return sentences

            

