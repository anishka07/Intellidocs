import os

from llms.gemini_response import gemini_response
from model.intellidocs_rag_final.intellidocs_main import IntellidocsRAG
from utils.constants import PathSettings, ConstantSettings

rag = IntellidocsRAG(
    chunk_size=ConstantSettings.CHUNK_SIZE,
    embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
    chroma_db_dir=PathSettings.CHROMA_DB_PATH
)
rag.process(pdf_doc_paths=[
        os.path.join(PathSettings.PDF_DIR_PATH, 'POM_Unit-1.pdf'),
    ],)

user_query = input("Enter query: ")

if __name__ == '__main__':
    # Step 1: Extract text
    extracted_texts = rag.extract_text_from_documents_fitz()

    # Step 2: Chunk text
    chunked_texts = rag.text_chunking(extracted_texts)

    extracted_texts_embeddings = rag.generate_embeddings(
        chunked_texts=chunked_texts,
        batch_size=16
    )
    # Step 4: Store embeddings (only if not already stored)
    rag.store_embeddings(
        chunked_texts=chunked_texts,
        embeddings_dict=extracted_texts_embeddings
    )
    print("Embeddings stored successfully.")

    # Step 5: Retrieve results using the correct pdf_key
    pdf_key = 'POM_Unit-1_key'
    top_results = rag.retrieve_top_n_custom(
        user_query=user_query,
        pdf_key=pdf_key,
        top_n=5
    )
    for result in top_results:
        print(result)

    # Step 6: Structure the response with a LLM
    llm_response = gemini_response(user_query=user_query, context=top_results)
    print(llm_response)
