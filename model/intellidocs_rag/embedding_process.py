import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from model.intellidocs_rag.intellidocs_rag_constants import id_rag_pdf_path, id_pdf_name, sent_tokenizer_model_name
from model.intellidocs_rag.pdf_loader import pdf_loader_main
from model.intellidocs_rag.chunk_processor import tp_main
from utils.constants import PathSettings


class EmbeddingProcessor:

    def __init__(self, embedding_model_name: str, pages_and_chunks: list[dict], project_dir: str):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.pages_and_chunks = pages_and_chunks
        self.project_dir = project_dir

    def move_model_to_device(self, device: str = "cpu"):
        """Moves the embedding model to the specified device (cpu or gpu)."""
        self.embedding_model.to(device)

    def encode_chunk(self, chunk: str):
        """Encodes a single chunk using the embedding model."""
        return self.embedding_model.encode(chunk)

    def add_embeddings_to_chunks(self):
        """Add embeddings for each chunk in the data and return updated pages_and_chunks."""
        for item in tqdm(self.pages_and_chunks):
            item["embeddings"] = self.encode_chunk(item["sentence_chunk"])
        return self.pages_and_chunks

    def encode_chunks_batch(self, batch_size: int = 32):
        """Encodes all chunks in batches and returns the embeddings as tensors."""
        text_chunks = [item["sentence_chunk"] for item in self.pages_and_chunks]
        text_chunk_embeddings = self.embedding_model.encode(
            text_chunks,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        return text_chunk_embeddings

    def save_embeddings_to_csv(self, save_dir: str = "CSV_db", csv_name: str = "test1.csv"):
        """Saves the processed chunks with embeddings into a CSV file."""
        text_chunks_and_embeddings_df = pd.DataFrame(self.pages_and_chunks)
        embeddings_df_save_path = os.path.join(self.project_dir, save_dir, csv_name)
        os.makedirs(os.path.dirname(embeddings_df_save_path), exist_ok=True)  # Ensure directory exists
        text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)
        print(f"Embeddings saved to {embeddings_df_save_path}")


if __name__ == '__main__':
    extracted_text_dict = pdf_loader_main(path=id_rag_pdf_path, pdf_name=id_pdf_name)
    chunks = tp_main(pgs_texts=extracted_text_dict, min_token_len=30)

    ep = EmbeddingProcessor(embedding_model_name=sent_tokenizer_model_name, pages_and_chunks=chunks, project_dir=PathSettings.PROJECT_DIR_PATH)
    ep.move_model_to_device(device="cpu")
    pages_and_chunks_with_embeddings = ep.add_embeddings_to_chunks()
    ep.save_embeddings_to_csv()
    print(pages_and_chunks_with_embeddings[0])


