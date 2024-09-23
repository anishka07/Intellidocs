import csv
import os
import pickle
import re

import fitz
import numpy as np
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm.auto import tqdm

from utils.constants import PathSettings


class PdfTextLoader:

    def __init__(self, pdf_name: str, pdf_path: str) -> None:
        self.pdf_name = pdf_name
        self.pdf_path = pdf_path
        self.complete_pdf_path = os.path.join(self.pdf_path, self.pdf_name)
        if not os.path.exists(self.complete_pdf_path):
            print("This file does not exists: ", self.complete_pdf_path)
        self.pdf_reader = PdfReader(self.complete_pdf_path)
        self.doc = fitz.open(self.pdf_path)

    def pdf_text_extractor(self) -> str:
        all_text = ""
        for page_number, page in tqdm(enumerate(doc), total=len(doc), desc="Extracting PDF"):
            text = page.get_text("text")
            all_text += text

        doc.close()

        all_text = all_text.replace("\n", " ").replace("\t", " ")

        return all_text

    def chunk_text(self, chunk_size: int = 1000, chunk_overlap: int = 20) -> list[str]:
        """
        Splits the input text into chunks of a specified size, with an overlap between chunks.

        :param chunk_size: Maximum size of each chunk.
        :param chunk_overlap: Number of overlapping words between consecutive chunks.
        :return: A list of text chunks.
        """
        extracted_text = self.pdf_text_extractor()

        def split_into_sentences(pdf_text: str = extracted_text) -> list[str]:
            # Split the text into sentences
            pdf_text = pdf_text.replace("\n", " ")
            return re.split(r'(?<=[.!?])\s+', pdf_text)

        def get_overlap_words(pdf_text: str = extracted_text) -> list[str]:
            # Get the last `chunk_overlap` words from a chunk
            words = pdf_text.split()
            return words[-chunk_overlap:] if len(words) > chunk_overlap else words

        sentences = split_into_sentences()
        text_chunks = []
        current_chunk = []
        current_chunk_size = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Check if the current chunk size with the next sentence exceeds the chunk size
            if current_chunk_size + sentence_length > chunk_size:
                # If so, add the current chunk to the list
                if current_chunk:
                    text_chunks.append(" ".join(current_chunk))
                    overlap_words = get_overlap_words(" ".join(current_chunk))
                    current_chunk = overlap_words  # Start a new chunk with the overlap words
                    current_chunk_size = len(" ".join(overlap_words))

            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_chunk_size += sentence_length

        # Add the last chunk if it has any sentences left
        if current_chunk:
            text_chunks.append(" ".join(current_chunk))

        return text_chunks


class TextEmbedding:

    def __init__(self, vector_size: int, min_count: int, epochs: int):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def train_doc2vec(self, text_chunks: list[str], save_model: bool, model_name: str) -> None:
        """
        Trains a Doc2Vec model on the provided text chunks.
        Args:
            text_chunks: list of text chunks.
            save_model: if true, saves the model in pickle file
            model_name: the name you want your model to be called (e.g. embeddings_v1.pkl)

        Returns:

        """
        # Prepare the training data by tagging each document (chunk)
        tagged_data = [TaggedDocument(words=chunk.split(), tags=[str(i)]) for i, chunk in enumerate(text_chunks)]

        # Initialize the Doc2Vec model
        self.model = Doc2Vec(vector_size=self.vector_size, min_count=self.min_count, epochs=self.epochs)

        # Build the vocabulary and train the model
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.epochs)

        if save_model:
            if not model_name.endswith(".pkl"):
                model_name += ".pkl"
            pickle_save_path = os.path.join(PathSettings.PICKLE_DIR_PATH, model_name)
            if not os.path.exists(pickle_save_path):
                with open(pickle_save_path, "wb") as f:
                    pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print("Pickle file for this embedding already exists.")

    def embed_text(self, text_chunks: list[str]) -> list[list[float]]:
        if not self.model:
            raise ValueError("Model must be trained before embedding. Call train_doc2vec() first.")
        text_embeddings = [self.model.infer_vector(chunk.split()) for chunk in text_chunks]
        return text_embeddings


def save_embeddings_to_csv(text_chunks, text_embeddings, csv_file_name: str):
    if not csv_file_name.endswith(".csv"):
        csv_file_name += ".csv"
    csv_save_path = os.path.join(PathSettings.CSV_DB_DIR_PATH, csv_file_name)
    if not os.path.exists(csv_save_path):
        with open(csv_save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Chunk_Index", "Text", "Embedding"])  # Header

            for idx, (chunk, embedding) in enumerate(zip(text_chunks, text_embeddings)):
                # Convert the embedding list to a string for storing in CSV
                embedding_str = ','.join(map(str, embedding))
                writer.writerow([f"Chunk_{idx + 1}", chunk, embedding_str])
        print("Csv file saved successfully at: ", csv_save_path)
    else:
        print("Csv file already exists at: ", csv_save_path)


def load_embeddings_from_csv(csv_file_name: str):
    if not csv_file_name.endswith(".csv"):
        csv_file_name += ".csv"
    csv_save_path = os.path.join(PathSettings.CSV_DB_DIR_PATH, csv_file_name)
    text_chunks = []
    text_embeddings = []
    if os.path.exists(csv_save_path):
        with open(csv_save_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skips the header
            for row in reader:
                chunked_text = row[1]
                embedding = np.array(list(map(float, row[2].split(','))))
                text_chunks.append(chunked_text)
                text_embeddings.append(embedding)

        return text_chunks, np.array(text_embeddings)


def find_most_similar_chunk(query_text, doc2vec_model, text_embeddings, text_chunks, top_n=1):
    # Embed the query
    query_embedding = doc2vec_model.infer_vector(query_text.split())

    # Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], text_embeddings)[0]

    # Get the indices of the top N most similar chunks
    top_n_indices = similarities.argsort()[-top_n:][::-1]

    return [(text_chunks[idx], similarities[idx]) for idx in top_n_indices]


if __name__ == '__main__':
    pdf_text_loader = PdfTextLoader(pdf_name='health.pdf', pdf_path=PathSettings.PDF_DIR_PATH)
    chunks = pdf_text_loader.chunk_text(chunk_size=1000, chunk_overlap=20)

    text_embedder = TextEmbedding(vector_size=100, min_count=2, epochs=100)
    text_embedder.train_doc2vec(text_chunks=chunks, save_model=False, model_name='embeddings_v4')
    embeddings = text_embedder.embed_text(text_chunks=chunks)

    save_embeddings_to_csv(text_chunks=chunks, text_embeddings=embeddings, csv_file_name='chunks_and_embeddings_v4')

    chunks, embeddings = load_embeddings_from_csv('chunks_and_embeddings_v4')

    # Simulate a user query
    query = "symptoms of pellagra"
    similar_chunks = find_most_similar_chunk(query, text_embedder.model, embeddings, chunks, top_n=3)

    for chunk, similarity in similar_chunks:
        print(f"Chunk: {chunk}")
        print(f"Similarity: {similarity}")
        print("-" * 50)
