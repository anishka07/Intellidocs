import hashlib
import logging
import os
import pickle
import uuid

import chromadb
import fitz
import spacy
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from utils.constants import ConstantSettings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class IntellidocsRAG:

    def __init__(self, pdf_doc_path: str, chunk_size: int, embedding_model: str, chroma_db_dir: str) -> None:
        self.pdf_path = pdf_doc_path
        self.chunk_size = chunk_size
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chroma_client = chromadb.Client(Settings(persist_directory=chroma_db_dir))
        self.cache_dir = os.path.join(chroma_db_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_pdf_hash(self) -> str:
        """Generate a unique hash for the PDF based on its content."""
        hasher = hashlib.md5()
        with open(self.pdf_path, "rb") as pdf_file:
            hasher.update(pdf_file.read())
        return hasher.hexdigest()

    def _load_cache(self, cache_key: str):
        """Load cached data for a given key."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as cache_file:
                return pickle.load(cache_file)
        return None

    def _save_cache(self, cache_key: str, data):
        """Save data to cache with a given key."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        with open(cache_path, "wb") as cache_file:
            pickle.dump(data, cache_file)

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
        pdf_hash = self._get_pdf_hash()
        cache_key = f"{pdf_hash}_chunks"
        cached_chunks = self._load_cache(cache_key)

        if cached_chunks:
            logger.info("Loaded text chunks from cache.")
            return cached_chunks

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

        self._save_cache(cache_key, chunks)
        return chunks

    def generate_embeddings(self, text_chunks: list[str], batch_size: int = 16) -> list[list[float]]:
        pdf_hash = self._get_pdf_hash()
        cache_key = f"{pdf_hash}_embeddings"
        cached_embeddings = self._load_cache(cache_key)

        if cached_embeddings:
            logger.info("Loaded embeddings from cache.")
            return cached_embeddings

        logger.info("Generating embeddings...")
        embeddings = []
        for i in tqdm(range(0, len(text_chunks), batch_size), desc="Generating embeddings in batches"):
            batch = text_chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch)  # Returns a numpy array
            embeddings.extend(batch_embeddings.tolist())  # Convert numpy array to list

        self._save_cache(cache_key, embeddings)
        return embeddings

    def store_embeddings(self, text_chunks: list[str], embeddings: list[list[float]], collection_name: str) -> None:
        logger.info("Storing embeddings...")
        try:
            collection = self.chroma_client.get_or_create_collection(name=collection_name)
            collection.add(
                documents=text_chunks,
                embeddings=embeddings,
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

