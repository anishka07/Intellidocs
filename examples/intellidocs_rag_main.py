import os
from model.intellidocs_rag_final.id_rag_updated import IntellidocsRAG
from model.llms.gemini_response import gemini_response
from utils.constants import PathSettings, ConstantSettings

rag = IntellidocsRAG(
    pdf_doc_paths=[
        os.path.join(PathSettings.PDF_DIR_PATH, 'POM_Unit-1.pdf'),
    ],
    chunk_size=400,
    embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
    chroma_db_dir=PathSettings.CHROMA_DB_PATH
)

user_query = "what is the concept of organization?"

if __name__ == '__main__':
    # Step 1: Extract text
    extracted_texts = rag.extract_text_from_documents_fitz()

    # Step 2: Chunk text
    chunked_texts = rag.text_chunking(extracted_texts)

    # Step 3: Generate embeddings
    extracted_texts_embeddings = rag.generate_embeddings(
        chunked_texts=chunked_texts,
        batch_size=16
    )
    #
    # Step 4: Store embeddings (only if not already stored)
    rag.store_embeddings(
        chunked_texts=chunked_texts,
        embeddings_dict=extracted_texts_embeddings
    )
    print("Embeddings stored successfully.")

    # Step 5: Retrieve results using the correct pdf_key
    pdf_key = 'POM_Unit-1_key'
    top_results = rag.retrieve_top_n(
        user_query=user_query,
        pdf_key=pdf_key,
        top_n=10
    )
    all_text = ''.join(text for text in top_results[0]['chunk'])
    print(all_text)
    # llm_response = gemini_response(user_query=user_query, context=all_text)
    # print(llm_response)
    # print()
