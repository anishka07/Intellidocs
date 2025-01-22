import logging
from pymilvus import utility
from model.intellidocs_rag_final.chunk_processor import tp_main
from model.intellidocs_rag_final.embedding_process_V1 import embedding_process_main
from model.intellidocs_rag_final.intellidocs_rag_constants import (
    id_rag_pdf_path, id_pdf_name, sent_tokenizer_model_name
)
from model.intellidocs_rag_final.milvus_conn_manager import MilvusConnectionManager
from model.intellidocs_rag_final.pdf_loader import pdf_loader_main
from model.intellidocs_rag_final.retrieval_processV1 import retriever_main
from model.llms.llama_response import setup_pipeline
from utils.constants import PathSettings, ConstantSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelliDocsProcessor:

    def __init__(self, model_name: str = ConstantSettings.LLM_MODEL_NAME):
        self.pipeline = setup_pipeline(model_name=model_name)
        self.connection_manager = MilvusConnectionManager()

    def process_document(self, user_query: str, collection_name: str, rag_device: str) -> str:
        """Main processing function with proper connection management"""
        try:
            # Ensure connections are established
            self.connection_manager.ensure_connections()
            logger.info(f"Processing query for collection: {collection_name}")
            # Check if collection exists
            if not utility.has_collection(collection_name, using="em"):
                logger.info(f"Creating new collection: {collection_name}")
                # Extract and process PDF
                pdf_text = pdf_loader_main(path=id_rag_pdf_path, pdf_name=id_pdf_name)
                chunks = tp_main(pgs_texts=pdf_text, min_token_len=30)
                # Create embeddings
                embedding_process_main(
                    embedding_model_name=sent_tokenizer_model_name,
                    pages_and_chunks=chunks,
                    project_dir=PathSettings.PROJECT_DIR_PATH,
                    collection_name=collection_name,
                    dev=rag_device,
                    connection_alias="em"
                )
                logger.info("Embeddings created successfully")
            # Retrieve results
            results = retriever_main(
                collection_name=collection_name,
                user_query=user_query
            )
            # Generate response
            context = "".join(r["text"] for r in results)
            return context
            # response = llama_response(
            #     query=user_query,
            #     context=context,
            #     p=self.pipeline
            # )
            # return response
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            raise

    def __del__(self):
        """Cleanup connections on object destruction"""
        self.connection_manager.disconnect_all()


def intelli_docs_main(user_query: str, collection_name: str, rag_device: str) -> str:
    """Main entry point with proper resource management"""
    processor = IntelliDocsProcessor()
    try:
        return processor.process_document(user_query, collection_name, rag_device)
    finally:
        # Ensure connections are cleaned up
        processor.connection_manager.disconnect_all()


if __name__ == '__main__':
    try:
        result = intelli_docs_main(
            user_query="what are macro nutrients?",
            collection_name="intellidocs_collection_v2",
            rag_device="cpu"
        )
        print("Generating Structured Response....")
        print(result)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
