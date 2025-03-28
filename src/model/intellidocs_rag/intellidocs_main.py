import os
import typing
from typing import Dict, List, Optional

import fitz
import spacy
from tqdm.auto import tqdm

from src.algorithms.cosine_similarity import cosine_similarity
from src.model.intellidocs_rag.intellidocs_rag_base import BaseIntelliDocs
from utils.constants import ConstantSettings, PathSettings


class IntellidocsRAG(BaseIntelliDocs):
    def __init__(
        self,
        chunk_size: int = ConstantSettings.CHUNK_SIZE,
        embedding_model: str = ConstantSettings.EMBEDDING_MODEL_NAME,
        chroma_db_dir: str = PathSettings.CHROMA_DB_PATH,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            embedding_model=embedding_model,
            chroma_db_dir=chroma_db_dir,
            cache_dir=cache_dir,
        )

    def _generate_document_key(self, document_path: str) -> str:
        """
        Generate a unique key for a PDF document using its filename.

        :param document_path: Path to the PDF document
        :return: Unique document key
        """
        return os.path.splitext(os.path.basename(document_path))[0] + "_key"

    def extract_text(self, document_paths: List[str]) -> Dict[str, Optional[str]]:
        """
        Extract text from PDF documents using PyMuPDF (Fitz).

        :param document_paths: List of PDF document paths
        :return: Dictionary of document keys to extracted text
        """
        extracted_texts = {}
        for pdf_path in document_paths:
            pdf_key = self.document_keys[pdf_path]
            cache_key = f"{pdf_key}_text"

            # Check cache first
            cached_text = self._load_cache(cache_key)
            if cached_text:
                self.logger.info(f"Loaded extracted text from cache for {pdf_path}.")
                extracted_texts[pdf_key] = cached_text
                continue

            try:
                self.logger.info(f"Extracting text from {pdf_path}...")
                docs = fitz.open(pdf_path)
                all_text = "".join(
                    page.get_text()
                    for page in tqdm(docs, desc=f"Extracting {pdf_path}")
                )
                docs.close()

                if not all_text.strip():
                    self.logger.warning(f"No text could be extracted from {pdf_path}.")
                    extracted_texts[pdf_key] = None
                else:
                    extracted_texts[pdf_key] = all_text
                    self._save_cache(cache_key, all_text)  # Cache extracted text

            except Exception as e:
                self.logger.error(f"Failed to extract text from {pdf_path}: {e}")
                extracted_texts[pdf_key] = None

        return extracted_texts

    def chunk_text(self, extracted_texts: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Chunk the extracted text for each PDF using Spacy for sentence segmentation.

        :param extracted_texts: Dictionary of document keys to full text
        :return: Dictionary of document keys to text chunks
        """
        chunked_texts = {}

        for pdf_key, extracted_text in extracted_texts.items():
            if extracted_text is None:
                continue

            cache_key = f"{pdf_key}_chunks"
            cached_chunks = self._load_cache(cache_key)

            if cached_chunks:
                self.logger.info(f"Loaded text chunks from cache for {pdf_key}.")
                chunked_texts[pdf_key] = cached_chunks
                continue

            self.logger.info(f"Chunking extracted text for {pdf_key}...")
            nlp = spacy.load(ConstantSettings.SPACY_LOAD)
            nlp.max_length = max(len(extracted_text), nlp.max_length)
            doc = nlp(extracted_text)
            sentences = [
                sent.text.strip() for sent in doc.sents if sent.text.strip()
            ]  # Remove empty sentences

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
                    current_chunk = [
                        sentence
                    ]  # Start a new chunk with the current sentence
                    current_length = sentence_length

            if current_chunk:  # Add the last chunk if it's not empty
                chunk_text = " ".join(current_chunk)
                if chunk_text not in chunks:  # Avoid duplicate chunks
                    chunks.append(chunk_text)

            self._save_cache(cache_key, chunks)
            chunked_texts[pdf_key] = chunks

        return chunked_texts

    def retrieve_top_n(
        self, user_query: str, doc_key: str, top_n: int = 5
    ) -> List[Dict[str, typing.Union[str, float]]]:
        """
        Manually compute cosine similarity and retrieve top N results for a specific document.

        :param user_query: User's search query
        :param doc_key: Key of the document collection to search
        :param top_n: Number of top results to return
        :return: List of dictionaries with chunk and similarity score
        """
        self.logger.info(
            f"Retrieving top {top_n} results for {doc_key} using custom cosine similarity..."
        )

        try:
            collection_name = f"pdf_{doc_key}"
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )

            query_embedding = self.embedding_model.encode([user_query]).tolist()[0]
            results = collection.get(include=["embeddings", "documents"])

            if not results["documents"]:
                return []

            similarities = []
            seen_chunks = set()  # Track seen chunks to avoid duplicates
            for doc_embedding, doc_text in zip(
                results["embeddings"], results["documents"]
            ):
                if doc_text not in seen_chunks:  # Skip duplicates
                    similarity = cosine_similarity(query_embedding, doc_embedding)
                    similarities.append({"chunk": doc_text, "score": similarity})
                    seen_chunks.add(doc_text)

            # Sort by similarity and return top N
            similarities.sort(key=lambda x: x["score"], reverse=True)
            return similarities[:top_n]

        except Exception as e:
            self.logger.error(f"Error retrieving results for {doc_key}: {e}")
            raise RuntimeError(f"Error retrieving results for {doc_key}: {e}")

    def process(self, pdf_doc_paths: List[str]) -> None:
        """
        Convenience method to maintain backwards compatibility with the original implementation.

        :param pdf_doc_paths: List of PDF document paths to process
        """
        self.process_documents(pdf_doc_paths)
