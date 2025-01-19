import os

import torch
from ollama import ChatResponse
from ollama import chat
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.intellidocs_rag_final.chunk_processor import tp_main
from model.intellidocs_rag_final.embedding_process import embedding_process_main
from model.intellidocs_rag_final.intellidocs_rag_constants import id_rag_pdf_path, id_pdf_name, \
    sent_tokenizer_model_name
from model.intellidocs_rag_final.pdf_loader import pdf_loader_main
from model.intellidocs_rag_final.retrieval_process import retriever_main
from utils.constants import PathSettings
import google.generativeai as genai


api_token = os.getenv("GOOGLE_GEMINI_API_TOKEN")
genai.configure(api_key=api_token)

query = "what are macro nutrients?"
#
# def load_llm(model_name):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token
#
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         pad_token_id=tokenizer.eos_token_id
#     )
#
#     return tokenizer, model


def intelli_docs_main(user_query: str, save_csv_name: str, save_csv_dir: str, rag_device: str) -> str:
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
        user_query=user_query
    )
    text_result = results['text']
    return text_result


def ollama_llm_response(pdf_results: str, query: str):
    response: ChatResponse = chat(model='mistral:latest', messages=[
        {
            'role': 'user',
            'content': f'You are a RAG and this is the query the user asked {query} and these are the top 5 responses. I want you to summarize this and give me a structured response: {pdf_results}',
        },
    ])
    return response['message']['content']


def gemini_llm_response(query: str):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        f'You are a RAG and this is the query the user asked {query} and these are the top 5 responses. I want you to summarize this and give me a structured response: {query}')
    return response.text


if __name__ == '__main__':
    result = intelli_docs_main(
        user_query=query,
        save_csv_name="mn.csv",
        save_csv_dir="CSV_db",
        rag_device="cpu"
    )
    print(result)
