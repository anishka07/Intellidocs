import os

from ollama import ChatResponse
from ollama import chat

from model.intellidocs_rag_final.chunk_processor import tp_main
from model.intellidocs_rag_final.embedding_process import embedding_process_main
from model.intellidocs_rag_final.intellidocs_rag_constants import id_rag_pdf_path, id_pdf_name, \
    sent_tokenizer_model_name
from model.intellidocs_rag_final.pdf_loader import pdf_loader_main
from model.intellidocs_rag_final.retrieval_process import retriever_main
from utils.constants import PathSettings


query = "what are macro nutrients?"


def intelli_docs_main(query: str, save_csv_name: str, save_csv_dir: str, rag_device: str):
    if not save_csv_name.endswith('.csv'):
        save_csv_name += '.csv'
    pdf_extracted_text = pdf_loader_main(path=id_rag_pdf_path, pdf_name=id_pdf_name)
    filtered_chunks = tp_main(pgs_texts=pdf_extracted_text, min_token_len=30)
    if embedding_process_main(
        embedding_model_name=sent_tokenizer_model_name,
        pages_and_chunks=filtered_chunks,
        project_dir=PathSettings.PROJECT_DIR_PATH,
        csv_name=save_csv_name,
        save_dir=save_csv_dir,
        dev=rag_device
    ):
        print("Embeddings created successfully.")
    results = retriever_main(
        embeddings_df_path=os.path.join(PathSettings.CSV_DB_DIR_PATH, save_csv_name),
        user_query=query
    )
    generated_response = llm_response(results)
    return generated_response


def llm_response(pdf_results: str):
    response: ChatResponse = chat(model='mistral:latest', messages=[
        {
            'role': 'user',
            'content': f'You are a RAG and this is the query the user asked {query} and these are the top 5 responses. I want you to summarize this and give me a structured response: {pdf_results}',
        },
    ])
    return response['message']['content']


if __name__ == '__main__':
    result = intelli_docs_main(
        query=query,
        save_csv_name="mn.csv",
        save_csv_dir="CSV_db",
        rag_device="cpu"
    )
    print(result)
