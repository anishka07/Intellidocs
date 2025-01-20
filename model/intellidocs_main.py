import os
import torch
from transformers import pipeline

from model.intellidocs_rag_final.chunk_processor import tp_main
from model.intellidocs_rag_final.embedding_process import embedding_process_main
from model.intellidocs_rag_final.intellidocs_rag_constants import id_rag_pdf_path, id_pdf_name, \
    sent_tokenizer_model_name
from model.intellidocs_rag_final.pdf_loader import pdf_loader_main
from model.intellidocs_rag_final.retrieval_process import retriever_main
from utils.constants import PathSettings

# api_token = os.getenv("GOOGLE_GEMINI_API_TOKEN")
# genai.configure(api_key=api_token)

query = "what are macro nutrients?"


def intelli_docs_main(user_query: str, save_csv_name: str, save_csv_dir: str, rag_device: str) -> str:
    if not save_csv_name.endswith('.csv'):
        save_csv_name += '.csv'

    save_csv_dir = os.path.abspath(save_csv_dir)  # Convert save_csv_dir to an absolute path
    csv_file_path = os.path.join(save_csv_dir, save_csv_name)  # Combine directory and filename

    print(f"Checking for CSV file at: {csv_file_path}")

    if not os.path.exists(csv_file_path):
        print(f"CSV file not found at {csv_file_path}. Creating embeddings...")
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
        else:
            raise Exception("Failed to create embeddings.")
    else:
        print(f"Using existing CSV file: {csv_file_path}")

    results = retriever_main(
        embeddings_df_path=csv_file_path,  # Use the resolved absolute path
        user_query=user_query
    )
    return results


pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])


if __name__ == '__main__':
    result = intelli_docs_main(
        user_query=query,
        save_csv_name="mn.csv",
        save_csv_dir="/Users/anishkamukherjee/Documents/Intellidocs/CSV_db",
        rag_device="cpu"
    )
    print(result)
    print(len(result))
