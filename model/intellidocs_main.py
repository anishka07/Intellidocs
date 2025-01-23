import os

from model.intellidocs_rag_final.chunk_processor import ChunkProcessor
from model.intellidocs_rag_final.embedding_process import EmbeddingProcessor
from model.intellidocs_rag_final.intellidocs_rag_constants import id_pdf_name
from model.intellidocs_rag_final.pdf_loader import PdfLoader
from model.intellidocs_rag_final.retrieval_process import Retriever
from utils.constants import PathSettings, ConstantSettings


def id_main(save_pdf_name: str, user_query: str, save_csv_name: str):
    embeddings_path = os.path.join(PathSettings.CSV_DB_DIR_PATH, save_csv_name)

    if os.path.exists(embeddings_path):
        print(f"Embeddings already exist at {embeddings_path}. Using existing embeddings.")
    else:
        pdf_loader = PdfLoader(
            pdf_path=os.path.join(PathSettings.PDF_DIR_PATH, id_pdf_name),
            save_pdf_name=save_pdf_name,  # Removed redundant path join
        )
        extracted_text_dict = pdf_loader.add_tokenized_sentences()

        chunk_processor = ChunkProcessor(pages_and_texts=extracted_text_dict, min_token_length=30)
        filtered_chunks = chunk_processor.filter_chunks_by_token_length()

        embedding_processor = EmbeddingProcessor(
            embedding_model_name=ConstantSettings.EMBEDDING_MODEL_NAME,
            pages_and_chunks=filtered_chunks,
            project_dir=PathSettings.PROJECT_DIR_PATH,
            csv_name=save_csv_name,  # Fixed csv_name alignment
            save_dir=PathSettings.CSV_DB_DIR_PATH  # Ensure directory consistency
        )
        embedding_processor.move_model_to_device(device="cpu")  # Simplified device argument
        embedding_processor.add_embeddings_to_chunks()
        embedding_processor.save_embeddings_to_csv()

    # Perform retrieval
    retriever = Retriever(
        embeddings_df_path=embeddings_path,
        model_name=ConstantSettings.EMBEDDING_MODEL_NAME,
    )
    scores, indices = retriever.retrieve_relevant_resources(user_query)

    results = []
    for score, index in zip(scores, indices):
        result = {
            "score": score.item(),
            "text": retriever.pages_and_chunks[index]["sentence_chunk"],
            "page_number": retriever.pages_and_chunks[index]["page_number"],
        }
        results.append(result)
    return results


if __name__ == '__main__':
    results = id_main(
        save_pdf_name=os.path.join(PathSettings.PDF_DIR_PATH, 'intellidocs_rag_final.pdf'),
        user_query="what are macronutrients?",
        save_csv_name='test.csv'
    )
    print(results)
