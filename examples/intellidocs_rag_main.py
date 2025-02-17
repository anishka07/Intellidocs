import os

from llms.gemini_response import gemini_response
from model.intellidocs_rag_final.intellidocs_main import IntellidocsRAG
from utils.constants import PathSettings, ConstantSettings

rag = IntellidocsRAG(
    chunk_size=ConstantSettings.CHUNK_SIZE,
    embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
    chroma_db_dir=PathSettings.CHROMA_DB_PATH
)

# Pre-process the document (only once)
pdf_path = os.path.join(PathSettings.PDF_DIR_PATH, 'POM_Unit-1.pdf')
rag.process(pdf_doc_paths=[pdf_path])

# Extract, chunk, and embed only once per document
print("Processing document...")
extracted_texts = rag.extract_text_from_documents_fitz()
chunked_texts = rag.text_chunking(extracted_texts)
extracted_texts_embeddings = rag.generate_embeddings(
    chunked_texts=chunked_texts, batch_size=16
)
rag.store_embeddings(
    chunked_texts=chunked_texts,
    embeddings_dict=extracted_texts_embeddings
)
print("Document processing completed.")

# Continuous user query loop
if __name__ == '__main__':
    while True:
        user_query = input("\nUser: ").strip()

        # Exit condition
        if user_query.lower() in ['quit', 'bye', 'exit']:
            print("Exiting IntelliDocs. Have a great day!")
            break

        # Retrieve results using the correct pdf_key
        pdf_key = 'POM_Unit-1_key'
        top_results = rag.retrieve_top_n_custom(
            user_query=user_query,
            pdf_key=pdf_key,
            top_n=5
        )

        # Get LLM response
        llm_response = gemini_response(user_query=user_query, context=top_results)
        print(f"IntelliDocs: {llm_response}")
