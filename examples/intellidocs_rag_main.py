import os

from model.intellidocs_rag_final.id_rag_updated import IntellidocsRAG
from utils.constants import PathSettings, ConstantSettings

rag = IntellidocsRAG(
    chunk_size=ConstantSettings.CHUNK_SIZE,
    embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
    chroma_db_dir=PathSettings.CHROMA_DB_PATH
)
rag.process(pdf_doc_paths=[
        os.path.join(PathSettings.PDF_DIR_PATH, 'monopoly.pdf'),
    ],)

user_query = "what are the monopoly speed die rules?"

if __name__ == '__main__':
    # Step 1: Extract text
    # extracted_texts = rag.extract_text_from_documents_fitz()
    extracted_texts1 = rag.extract_text_from_documents_pdfreader()

    # Step 2: Chunk text
    chunked_texts = rag.text_chunking(extracted_texts1)

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
    pdf_key = 'monopoly_key'
    try:
        top_results = rag.retrieve_top_n(
            user_query=user_query,
            pdf_key=pdf_key,
            top_n=10
        )

        if not top_results:
            print("No relevant chunks found.")
        else:
            top_texts = []
            for item in top_results:
                chunk = item['chunk'].strip()
                score = item['score']
                top_texts.append({
                    'text': chunk,
                    'score': score
                })

            print("\nRetrieved Chunks (ordered by relevance):")
            print("-" * 80)
            for i, item in enumerate(top_texts, 1):
                print(f"\nChunk {i} (Similarity Score: {item['score']:.4f}):")
                print(item['text'])
                print("-" * 80)

    except Exception as e:
        print(f"Error retrieving results: {e}")
    # top_results = rag.retrieve_top_n(
    #     user_query=user_query,
    #     pdf_key=pdf_key,
    #     top_n=5,
    # )
    # print(top_results)



