import os

from model.intellidocs_rag_final.chunk_processor import tp_main
from model.intellidocs_rag_final.embedding_process import embedding_process_main
from model.intellidocs_rag_final.intellidocs_rag_constants import id_rag_pdf_path, id_pdf_name, \
    sent_tokenizer_model_name
from model.intellidocs_rag_final.pdf_loader import pdf_loader_main
from model.intellidocs_rag_final.retrieval_process import retriever_main
from utils.constants import PathSettings


def intelli_docs_main(user_query: str, save_csv_name: str, save_csv_dir: str, rag_device: str) -> str:
    """

    :param user_query: users query
    :param save_csv_name: name of the csv file to be saved
    :param save_csv_dir: directory to save the csv file
    :param rag_device: rag device
    :return:
    """
    if not save_csv_name.endswith('.csv'):
        save_csv_name += '.csv'

    save_csv_dir = os.path.abspath(save_csv_dir)
    csv_file_path = os.path.join(save_csv_dir, save_csv_name)

    print(f"Checking for CSV file at: {csv_file_path}")

    if not os.path.exists(csv_file_path):
        print(f"CSV file not found at {csv_file_path}. Creating embeddings...")
        pdf_extracted_text = pdf_loader_main(path=id_rag_pdf_path, pdf_name=id_pdf_name)
        filtered_chunks = tp_main(pgs_texts=pdf_extracted_text, min_token_len=30)

        success = embedding_process_main(
            embedding_model_name=sent_tokenizer_model_name,
            pages_and_chunks=filtered_chunks,
            project_dir=PathSettings.PROJECT_DIR_PATH,
            csv_name=save_csv_name,
            save_dir=save_csv_dir,
            dev=rag_device
        )

        if not success:
            raise Exception("Failed to create embeddings.")
        print("Embeddings created successfully.")
    else:
        print(f"Using existing CSV file: {csv_file_path}")

    results = retriever_main(
        embeddings_df_path=csv_file_path,
        user_query=user_query
    )
    total_responses = ""
    for r in results:
        total_responses += r["text"]
    return total_responses


if __name__ == '__main__':
    result = intelli_docs_main(
        user_query="what is statistical learning?",
        save_csv_name="mn.csv",
        save_csv_dir="/Users/anishkamukherjee/Documents/Intellidocs/CSV_db",
        rag_device="cpu"
    )
    print("Generating Structured Response....")
    print(result)
