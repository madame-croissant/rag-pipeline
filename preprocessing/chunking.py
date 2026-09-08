from preprocessing.loader import DataLoader
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ChunkStrategies:
    def __init__(self):
        self.loader = DataLoader()
        self.loader.build_corpus('data/raw')

        #model for semantic split
        print("Loading SentenceTransformer model")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")


    def chunk_handler(self, strategy_func, strategy_name, **kwargs):

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
                #print(final_chunked_text)
        
        print("Total chunks after splitting:", len(final_chunked_text))

        
        #print(final_chunked_text)
        return final_chunked_text

    def fixed_size(self, text, size, overlap):

        chunks = []
        start = 0

        while start < len(text):
            end = start + size
            piece = text[start : end]

            chunks.append(piece)

            start = start + (size - overlap)

        return chunks
    
    def recursive_split(self, text, size, separators: list):

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
    
    def semantic_split(self, text, threshold, min_chunk_size):
        
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

    def split_into_sentences(self, text):
        """
        helper for semantic split
        """

        raw_sent = re.split(r'(?<=[.!?])\s+', text)

        sentences = [s.strip() for s in raw_sent if s.strip()]

        return sentences

            



if __name__ == "__main__":
    chunks = ChunkStrategies()
    #chunks.chunk_handler(chunks.fixed_size, "fixed", size=500, overlap=50)
    #chunks.chunk_handler(chunks.recursive_split, "recursive", size=500, separators=["\n\n", "\n", ".", " "])

    semantic_chunks = chunks.chunk_handler( chunks.semantic_split, "semantic", threshold=0.35, min_chunk_size=300)

    # Calculate character length of each chunk
    lengths = [len(c["text"]) for c in semantic_chunks]

    avg_len = sum(lengths) / len(lengths)
    print(f"Average Chunk Length: {avg_len:.1f} characters (~{avg_len / 5:.1f} words)")
    print("Shortest 3 chunks:", sorted(lengths)[:3])
    print("Sample chunk:\n", semantic_chunks[10]["text"])
            

