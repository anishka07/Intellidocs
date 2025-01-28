import os

from model.intellidocs_rag_final.id_chroma_rag import IntellidocsRAG
from model.intellidocs_rag_v3.intellidocs_rag_constants import id_pdf_name
from utils.constants import PathSettings, ConstantSettings

if __name__ == '__main__':
    idr = IntellidocsRAG(
        pdf_doc_path=os.path.join(PathSettings.PDF_DIR_PATH, id_pdf_name),
        chunk_size=ConstantSettings.CHUNK_SIZE,
        embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
        chroma_db_dir=PathSettings.CHROMA_DB_PATH
    )

    text = idr.extract_text_from_document_fitz()
    chunks = idr.text_chunking(text)

    embeddings = idr.generate_embeddings(chunks)
    idr.store_embeddings(chunks, embeddings, collection_name=ConstantSettings.CHROMA_DB_COLLECTION)

    query = input("Enter your query (quit to quit): ")
    while True:
        query = input("Enter your query (type 'quit' to exit): ")
        if query.lower() == "quit":
            print("Exiting. Goodbye!")
            break

        try:
            results = idr.retrieve_top_n(
                user_query=query,
                chroma_collection_name=ConstantSettings.CHROMA_DB_COLLECTION,
                top_n=5,
            )
            if not results:
                print("No results found for your query.")
            else:
                print("Top Result:")
                print(f"Chunk: {results[0]['chunk']}")
                print(f"Score: {results[0]['score']}")
        except Exception as e:
            print(f"An error occurred while retrieving results: {e}")


