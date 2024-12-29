from sentence_transformers import SentenceTransformer

class EmbeddingModel():
    def __init__(self):
        self.embedding_model = SentenceTransformer("Cloyne/vietnamese-embedding_finetuned")
        
    def get_embedding(self, text: str):
        if not text.strip():
            return []

        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
