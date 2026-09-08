import chromadb
from chromadb.utils import embedding_functions 

class VectoreStore:
    def __init__(self, db_path="./chroma_db", collection_name="rag_chunks"):
        self.client = chromadb.PersistentClient(path=db_path)
        
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
    

    def add_chunks(self, chunks: list[dict]):

        dist_threshold = 0.05 #(similarity 0.95, so 1-0.95

        unique_chunks =[]
        skipped_count = 0
        #documents = []
        #metadata = []
        #ids = []

        for idx, chunk in enumerate(chunks):
            #documents.append(chunk["text"])

            text= chunk["text"]

            if self.collection.count() > 0:
                query_res = self.collection.query(
                    query_texts=[text],
                    n_results=1
                )

                #print("This is query res", query_res)

                #checking the distance
                if query_res["distances"] and query_res["distances"][0]:
                    top_distance = query_res["distances"][0][0]

                    
                    #print(f"Chunk {idx} top distance: {top_distance:.4f}")

                    #dist < 0.05 -> same
                    if top_distance < dist_threshold:
                        skipped_count += 1
                        continue 

            chunk_id = f"chunk_{len(unique_chunks)}"
            
            chunk_metadata = {
                "source_file": chunk.get("source_file", "unknown"),
                "heading": chunk.get("heading", "none"),
                "chunking_strategy": chunk.get("chunking_strategy", "unknown"),
                "chunk_idx": idx,
                "character_count": len(text)
            }

            #ids.append(f"chunk_{idx}")
            
            
            
            self.collection.add(
                documents=[text],
                metadatas=[chunk_metadata],
                ids=[chunk_id]
            )

            unique_chunks.append(chunk)
        
        print(f"\n Deduplication")
        print(f"Total chunks: {len(chunks)}")
        print(f"Duplicates skipped: {skipped_count}")
        print(f"Unique chunks stored: {len(unique_chunks)}")
        
        return unique_chunks
    
        #batching
       

    def search_dense(self, query: str, k: int) -> list:

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        dense_results = []

        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] #[0] means a list of results for query 1, bc you can have multiple queries
            distances = results["distances"][0]

            for doc, meta, dist in zip(documents, metadatas, distances):

                similarity = 1.0 - dist
                dense_results.append({
                    "text": doc,
                    "metadata": meta,
                    "score": similarity,
                    "source": "dense"
                })

        #print("Dense results", dense_results)
        return dense_results
