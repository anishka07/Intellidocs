import logging
import os
import uuid

import chromadb
import fitz
import numpy as np
import spacy
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from model.intellidocs_rag_final.intellidocs_rag_constants import id_pdf_name
from utils.constants import PathSettings, ConstantSettings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class IntellidocsRAG:

    def __init__(self, pdf_doc_path: str, chunk_size: int, embedding_model: str, chroma_db_dir: str, collection_name: str) -> None:
        self.pdf_path = pdf_doc_path
        self.chunk_size = chunk_size
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, persist_directory=chroma_db_dir))

    def extract_texts_from_document_pdf_reader(self) -> str:
        from pypdf import PdfReader  # Lazy import. Only executed when called.
        all_text = ""
        reader = PdfReader(self.pdf_path)
        for page in tqdm(reader.pages):
            text = page.extract_text()
            all_text += text
        return all_text

    def extract_text_from_document_fitz(self) -> str:
        try:
            logger.info("Extracting text from document...")
            docs = fitz.open(self.pdf_path)
            all_text = "".join(page.get_text() for page in tqdm(docs, desc="Extracting text"))
            docs.close()
            if not all_text.strip():
                raise ValueError("No text could be extracted from the PDF.")
            return all_text
        except Exception as e:
            raise RuntimeError(f"Failed to extract text using fitz: {e}")

    def text_chunking(self, extracted_text: str) -> list[str]:
        logger.info("Chunking the extracted text into chunks...")
        nlp = spacy.load(ConstantSettings.SPACY_LOAD)

        nlp.max_length = max(len(extracted_text), nlp.max_length)
        doc = nlp(extracted_text)
        sentences = [sent.text for sent in doc.sents]

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in tqdm(sentences):
            if current_length + len(sentence) <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def generate_embeddings(self, text_chunks: list[str], batch_size: int = 16) -> list[list[float]]:
        logger.info("Generating embeddings...")
        embeddings = []
        for i in tqdm(range(0, len(text_chunks), batch_size), desc="Generating embeddings in batches"):
            batch = text_chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch)  # Returns a numpy array
            embeddings.extend(batch_embeddings.tolist())  # Convert numpy array to list
        return embeddings

    def store_embeddings(self, text_chunks: list[str], embeddings: list[list[float]]) -> None:
        logger.info("Storing embeddings...")
        try:
            collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
            collection.add(
                documents=text_chunks,
                embeddings=[embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in
                            embeddings],
                ids=[str(uuid.uuid4()) for _ in text_chunks],  # Generate unique IDs
            )
            logger.info("Embeddings stored successfully.")
        except Exception as e:
            raise RuntimeError(f"Error storing embeddings in ChromaDB: {e}")

    def retrieve_top_n(self, user_query: str, chroma_collection_name: str, top_n: int = 5) -> list[dict]:
        logger.info("Retrieving top n results...")
        try:
            collection = self.chroma_client.get_or_create_collection(chroma_collection_name)
            query_embedding = self.embedding_model.encode([user_query]).tolist()[0]
            results = collection.query(query_embeddings=[query_embedding], n_results=top_n)

            if not results["documents"]:
                return []

            return [
                {"chunk": doc, "score": score}
                for doc, score in zip(results["documents"], results["distances"])
            ]
        except Exception as e:
            raise RuntimeError(f"Error retrieving top N results: {e}")


if __name__ == '__main__':
    chroma_dir = os.path.join(PathSettings.CHROMA_DB_PATH)
    pdf_path = os.path.join(PathSettings.PDF_DIR_PATH, id_pdf_name)
    collection_name = "intellidocs_collection1"

    retriever = IntellidocsRAG(
        pdf_doc_path=pdf_path,
        chunk_size=150,
        embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
        chroma_db_dir=chroma_dir,
        collection_name=collection_name
    )

    # Extract and chunk text
    text = retriever.extract_text_from_document_fitz()
    chunks = retriever.text_chunking(text)

    # Generate embeddings and store them
    embeddings = retriever.generate_embeddings(chunks)
    retriever.store_embeddings(chunks, embeddings)

    # Retrieve top results for a query
    query = "What are macro nutrients?"
    results = retriever.retrieve_top_n(query, collection_name, top_n=5)

    print("Top results:")
    for result in results:
        print(result)
