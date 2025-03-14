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

from algorithms.cosine_similarity import cosine_similarity
from utils.constants import ConstantSettings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class IntellidocsRAG:

    def __init__(self, chunk_size: int, embedding_model: str, chroma_db_dir: str) -> None:
        self.chunk_size = chunk_size
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chroma_client = chromadb.Client(Settings(persist_directory=chroma_db_dir, anonymized_telemetry=False))
        self.cache_dir = os.path.join(chroma_db_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.pdf_keys = {}

    def process(self, pdf_doc_paths: list[str]):
        self.pdf_keys = {pdf_path: self._get_pdf_key(pdf_path) for pdf_path in pdf_doc_paths}  # Generate unique keys
        extracted_texts = self.extract_text_from_documents_fitz()
        chunked_texts = self.text_chunking(extracted_texts)
        extracted_texts_embeddings = self.generate_embeddings(chunked_texts)
        self.store_embeddings(chunked_texts, extracted_texts_embeddings)

    def get_pdf_keys(self):
        return self.pdf_keys.keys()

    def _get_pdf_key(self, pdf_path: str) -> str:
        """Generate a unique key for each PDF using its filename (without extension)."""
        return os.path.splitext(os.path.basename(pdf_path))[0] + "_key"  # Extract filename without extension

    def _load_cache(self, pdf_key: str):
        """Load cached data for a given PDF key."""
        cache_path = os.path.join(self.cache_dir, f"{pdf_key}.pkl")
        if os.path.exists(cache_path):
            logger.info(f"Cache data exists. Loading from {cache_path}")
            with open(cache_path, "rb") as cache_file:
                return pickle.load(cache_file)
        return None

    def _save_cache(self, pdf_key: str, data):
        """Save data to cache with a given PDF key."""
        cache_path = os.path.join(self.cache_dir, f"{pdf_key}.pkl")
        if os.path.exists(cache_path):
            logger.info(f"Cache data exists. Loading from {cache_path}")
            return
        with open(cache_path, "wb") as cache_file:
            pickle.dump(data, cache_file)

    def extract_text_from_documents_fitz(self) -> dict:
        """Extract text from multiple PDFs and store it using unique keys."""
        extracted_texts = {}
        for pdf_path, pdf_key in self.pdf_keys.items():
            cache_key = f"{pdf_key}_text"
            cached_text = self._load_cache(cache_key)

            if cached_text:
                logger.info(f"Loaded extracted text from cache for {pdf_path}.")
                extracted_texts[pdf_key] = cached_text
                continue

            try:
                logger.info(f"Extracting text from {pdf_path}...")
                docs = fitz.open(pdf_path)
                all_text = "".join(page.get_text() for page in tqdm(docs, desc=f"Extracting {pdf_path}"))
                docs.close()

                if not all_text.strip():
                    logger.warning(f"No text could be extracted from {pdf_path}.")
                    extracted_texts[pdf_key] = None
                else:
                    extracted_texts[pdf_key] = all_text
                    self._save_cache(cache_key, all_text)  # Cache extracted text

            except Exception as e:
                logger.error(f"Failed to extract text from {pdf_path}: {e}")
                extracted_texts[pdf_key] = None

        return extracted_texts

    def text_chunking(self, extracted_texts: dict) -> dict:
        """Chunk the extracted text for each unique PDF key."""
        chunked_texts = {}

        for pdf_key, extracted_text in extracted_texts.items():
            if extracted_text is None:
                continue

            cache_key = f"{pdf_key}_chunks"
            cached_chunks = self._load_cache(cache_key)

            if cached_chunks:
                logger.info(f"Loaded text chunks from cache for {pdf_key}.")
                chunked_texts[pdf_key] = cached_chunks
                continue

            logger.info(f"Chunking extracted text for {pdf_key}...")
            nlp = spacy.load(ConstantSettings.SPACY_LOAD)
            nlp.max_length = max(len(extracted_text), nlp.max_length)
            doc = nlp(extracted_text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]  # Remove empty sentences

            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in tqdm(sentences, desc=f"Chunking {pdf_key}"):
                sentence_length = len(sentence)
                if current_length + sentence_length <= self.chunk_size:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    if current_chunk:  # Ensure we don't add empty chunks
                        chunk_text = " ".join(current_chunk)
                        if chunk_text not in chunks:  # Avoid duplicate chunks
                            chunks.append(chunk_text)
                    current_chunk = [sentence]  # Start a new chunk with the current sentence
                    current_length = sentence_length

            if current_chunk:  # Add the last chunk if it's not empty
                chunk_text = " ".join(current_chunk)
                if chunk_text not in chunks:  # Avoid duplicate chunks
                    chunks.append(chunk_text)

            self._save_cache(cache_key, chunks)
            chunked_texts[pdf_key] = chunks

        return chunked_texts

    def generate_embeddings(self, chunked_texts: dict, batch_size: int = 16) -> dict:
        """Generate embeddings for text chunks and store them using unique keys."""
        embeddings_dict = {}

        for pdf_key, text_chunks in chunked_texts.items():
            cache_key = f"{pdf_key}_embeddings"
            cached_embeddings = self._load_cache(cache_key)

            if cached_embeddings:
                logger.info(f"Loaded embeddings from cache for {pdf_key}.")
                embeddings_dict[pdf_key] = cached_embeddings
                continue

            logger.info(f"Generating embeddings for {pdf_key}...")
            embeddings = []
            for i in tqdm(range(0, len(text_chunks), batch_size), desc=f"Embedding {pdf_key}"):
                batch = text_chunks[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(batch)
                embeddings.extend(batch_embeddings.tolist())

            self._save_cache(cache_key, embeddings)
            embeddings_dict[pdf_key] = embeddings

        return embeddings_dict

    def store_embeddings(self, chunked_texts: dict, embeddings_dict: dict) -> None:
        """Store embeddings in ChromaDB using unique PDF keys."""
        logger.info("Storing embeddings...")
        try:
            for pdf_key, text_chunks in chunked_texts.items():
                collection_name = f"pdf_{pdf_key}"
                collection = self.chroma_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None
                )

                # Check for existing documents in the collection
                existing_documents = collection.get(include=["documents"])["documents"]

                # Only add chunks that are not already in the collection
                unique_chunks = []
                unique_embeddings = []
                for chunk, embedding in zip(text_chunks, embeddings_dict[pdf_key]):
                    if chunk not in existing_documents:
                        unique_chunks.append(chunk)
                        unique_embeddings.append(embedding)

                if unique_chunks:
                    collection.add(
                        documents=unique_chunks,
                        embeddings=unique_embeddings,
                        ids=[str(uuid.uuid4()) for _ in unique_chunks],
                    )
                    logger.info(f"Stored {len(unique_chunks)} new embeddings for {pdf_key}.")
                else:
                    logger.info(f"No new chunks to store for {pdf_key}.")

        except Exception as e:
            raise RuntimeError(f"Error storing embeddings in ChromaDB: {e}")

    def retrieve_top_n_custom(self, user_query: str, pdf_key: str, top_n: int = 5) -> list[dict]:
        """Manually compute cosine similarity and retrieve top N results."""
        logger.info(f"Retrieving top {top_n} results for {pdf_key} using custom cosine similarity...")

        try:
            collection_name = f"pdf_{pdf_key}"
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None
            )

            query_embedding = self.embedding_model.encode([user_query]).tolist()[0]
            results = collection.get(include=["embeddings", "documents"])

            if not results["documents"]:
                return []

            # Compute cosine similarity for each document
            similarities = []
            seen_chunks = set()  # Track seen chunks to avoid duplicates
            for doc_embedding, doc_text in zip(results["embeddings"], results["documents"]):
                if doc_text not in seen_chunks:  # Skip duplicates
                    similarity = cosine_similarity(query_embedding, doc_embedding)
                    similarities.append({"chunk": doc_text, "score": similarity})
                    seen_chunks.add(doc_text)

            # Sort by similarity and return top N
            similarities.sort(key=lambda x: x["score"], reverse=True)
            return similarities[:top_n]

        except Exception as e:
            logger.error(f"Error retrieving results for {pdf_key}: {e}")
            raise RuntimeError(f"Error retrieving results for {pdf_key}: {e}")
