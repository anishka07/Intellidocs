import logging
import os
import pickle
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


class BaseIntelliDocs(ABC):

    def __init__(
            self,
            chunk_size: int,
            embedding_model: str,
            chroma_db_dir: str,
            cache_dir: Optional[str] = None
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.chunk_size = chunk_size
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chroma_client = chromadb.Client(
            Settings(
                persist_directory=chroma_db_dir,
                anonymized_telemetry=False
            )
        )
        self.cache_dir = cache_dir or os.path.join(chroma_db_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.document_keys = {}

    def _get_cache_path(self, key: str, suffix: str = "") -> str:
        filename = f"{key}{f'_{suffix}' if suffix else ''}.pkl"
        return os.path.join(self.cache_dir, filename)

    def _load_cache(self, key: str, suffix: str = "") -> Optional[object]:
        cache_path = self._get_cache_path(key, suffix=suffix)
        if os.path.exists(cache_path):
            self.logger.info(f"Loading cache from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, key: str, data: object, suffix: str = "") -> None:
        cache_path = self._get_cache_path(key, suffix=suffix)
        if not os.path.exists(cache_path):
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            self.logger.info(f"Saving cache to {cache_path}")

    @abstractmethod
    def _generate_document_key(self, document_path: str) -> str:
        pass

    @abstractmethod
    def extract_text(self, document_path: List[str]) -> Dict[str, Optional[str]]:
        pass

    @abstractmethod
    def chunk_text(self, extracted_text: Dict[str, str]) -> Dict[str, List[str]]:
        pass

    def generate_embeddings(
            self,
            chunked_texts: Dict[str, List[str]],
            batch_size: int = 16,
    ) -> Dict[str, List[List[float]]]:
        embeddings_dict = {}
        for doc_key, text_chunks in chunked_texts.items():
            cache_key = f"{doc_key}_embeddings"
            cached_embeddings = self._load_cache(cache_key)
            if cached_embeddings:
                self.logger.info(f"Loading embedding for {doc_key}")
                embeddings_dict[doc_key] = cached_embeddings
                continue
            self.logger.info(f"Generating embedding for {doc_key}")
            embeddings = []
            for i in tqdm(range(0, len(text_chunks), batch_size), desc=f"Generating embeddings for {doc_key}"):
                batch = text_chunks[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(batch)
                embeddings.extend(batch_embeddings.tolist())

            self._save_cache(cache_key, embeddings)
            embeddings_dict[doc_key] = embeddings
        return embeddings_dict

    def store_embeddings(
            self,
            chunked_texts: Dict[str, List[str]],
            embeddings_dict: Dict[str, List[List[float]]]
        ) -> None:
        self.logger.info("Storing embeddings")
        for doc_key, text_chunks in chunked_texts.items():
            collection_name = f"pdf_{doc_key}"
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None
            )
            existing_documents = collection.get(
                include=["documents"]
            )["documents"]
            unique_chunks = []
            unique_embeddings = []
            for chunk, embedding in zip(text_chunks, embeddings_dict[doc_key]):
                if chunk not in existing_documents:
                    unique_chunks.append(chunk)
                    unique_embeddings.append(embedding)
            if unique_chunks:
                collection.add(
                    documents=unique_chunks,
                    embeddings=unique_embeddings,
                    ids=[str(uuid.uuid4()) for _ in unique_chunks],
                )
                self.logger.info(f"Stored {len(unique_chunks)} new embeddings for {doc_key}")
            else:
                self.logger.info(f"No new chunks to store for {doc_key}")

    @abstractmethod
    def retrieve_top_n(
            self,
            user_query: str,
            doc_key: str,
            top_n: int = 5
    ) -> List[Dict[str, float]]:
        pass

    def process_documents(self, document_paths: List[str]) -> None:
        self.document_keys = {
            doc_path: self._generate_document_key(doc_path)
            for doc_path in document_paths
        }
        extracted_texts = self.extract_text(document_paths)
        chunked_texts = self.chunk_text(extracted_texts)
        embeddings_dict = self.generate_embeddings(chunked_texts)
        self.store_embeddings(chunked_texts, embeddings_dict)
