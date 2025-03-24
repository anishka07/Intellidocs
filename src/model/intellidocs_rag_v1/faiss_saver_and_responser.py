import os
from typing import List

import faiss
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from model.intellidocs_rag_v1.document_processor import PDFProcessor
from model.intellidocs_rag_v1.text_processor import TextProcessor
from utils.constants import PathSettings, ConstantSettings

load_dotenv()

GA_API_KEY = os.getenv("GOOGLE_GEMINI_API_TOKEN")
genai.configure(api_key=GA_API_KEY)


class VectorStore:

    def __init__(self, model_name: str = ConstantSettings.EMBEDDING_MODEL_NAME):
        """
        Initializes the VectorStore class to embed and store text chunks in FAISS.

        Args:
            model_name (str): The name of the pre-trained sentence transformer model for embeddings.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.text_chunks = []

        # Initialize Gemini model
        self.gemini_model = genai.GenerativeModel('gemini-pro')

    def embed_text_chunks(self, text_chunks: List[List[str]]) -> np.ndarray:
        """
        Embeds the text chunks using a sentence transformer model.

        Args:
            text_chunks (list[list[str]]): List of text chunks to embed.

        Returns:
            np.ndarray: An array of embedded vectors.
        """
        self.text_chunks = text_chunks
        concatenated_chunks = [' '.join(chunk) for chunk in text_chunks]
        embeddings = self.model.encode(concatenated_chunks)
        return np.array(embeddings)

    def create_faiss_index(self, embedding_size: int):
        """
        Creates a FAISS index based on the embedding size.

        Args:
            embedding_size (int): Size of the embedding vectors.
        """
        self.index = faiss.IndexFlatL2(embedding_size)

    def add_embeddings_to_index(self, embeddings: np.ndarray):
        """
        Adds embedded vectors to the FAISS index.

        Args:
            embeddings (np.ndarray): Embedding vectors to add to the index.
        """
        if self.index is None:
            raise ValueError("FAISS index is not created. Call create_faiss_index first.")
        self.index.add(embeddings)

    def search(self, query_text: str, top_k: int = 5):
        """
        Searches for similar vectors in the FAISS index and retrieves corresponding text chunks.

        Args:
            query_text (str): Query text to search for.
            top_k (int, optional): The number of top similar results to return. Defaults to 5.

        Returns:
            str: The generated response from Gemini.
        """
        query_embedding = self.model.encode([query_text])
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve text chunks based on indices
        relevant_chunks = [self.text_chunks[idx] for idx in indices[0]]
        concatenated_chunks = " ".join([" ".join(chunk) for chunk in relevant_chunks])

        # Use Gemini to generate a response
        prompt = f"Based on the following information, please answer the query: '{query_text}'\n\nContext: {concatenated_chunks}"
        response = self.gemini_model.generate_content(prompt)

        return response.text


if __name__ == '__main__':
    pdf_processor = PDFProcessor("nlp.pdf", PathSettings.PDF_DIR_PATH)
    extracted_text = pdf_processor.extract_text()
    cleaned_text = pdf_processor.clean_text()

    text_processor = TextProcessor(cleaned_text, 100)
    tokenized_sentences = text_processor.sentence_tokenizer()
    chunks = text_processor.split_text_to_chunks()

    vector_store = VectorStore()
    embeddings = vector_store.embed_text_chunks(chunks)

    # Create FAISS index and add embeddings
    vector_store.create_faiss_index(embedding_size=embeddings.shape[1])
    vector_store.add_embeddings_to_index(embeddings)

    # Query the index (example)
    query_text = "what does this book say about Data collection and labeling?"
    gemini_response = vector_store.search(query_text, top_k=5)

    print('Gemini response:', gemini_response)
