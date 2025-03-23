import os

from llms.gemini_response import gemini_response
from model.intellidocs_rag.intellidocs_main import IntellidocsRAG
from utils.constants import PathSettings, ConstantSettings

rag = IntellidocsRAG(
    chunk_size=ConstantSettings.CHUNK_SIZE,
    embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
    chroma_db_dir=PathSettings.CHROMA_DB_PATH
)

pdf_path = os.path.join(PathSettings.PDF_DIR_PATH, 'POM_Unit-1.pdf')
pdf_path1 = os.path.join(PathSettings.PDF_DIR_PATH, 'monopoly.pdf')
rag.process(pdf_doc_paths=[pdf_path, pdf_path1])

extracted_texts = rag.extract_text_from_documents_fitz()
chunked_texts = rag.text_chunking(extracted_texts)
extracted_texts_embeddings = rag.generate_embeddings(
    chunked_texts=chunked_texts, batch_size=16
)
rag.store_embeddings(
    chunked_texts=chunked_texts,
    embeddings_dict=extracted_texts_embeddings
)


def query_pdf(document_pdf_key: str, query_from_user: str):
    indexed_keys = list(rag.pdf_keys.values())
    if document_pdf_key not in indexed_keys:
        print("PDF key '{}' not found.".format(document_pdf_key))
        return False
    else:
        top_res = rag.retrieve_top_n_custom(
            user_query=query_from_user,
            pdf_key=document_pdf_key,
            top_n=5
        )
        llm_res = gemini_response(
            user_query=query_from_user,
            context=top_res,
        )
        print("IntelliDocs: ", llm_res)
        return True


if __name__ == '__main__':
    indexed_keys = list(rag.pdf_keys.values())

    if not indexed_keys:
        print("No PDF keys found.")
    else:
        print("Indexed PDF keys: ", list(rag.pdf_keys.values()))
        pdf_key = input("Enter PDF key to retrieve: ").strip()
        if pdf_key not in indexed_keys:
            print("PDF key '{}' not found.".format(pdf_key))
        else:
            while True:
                user_query = input("\nUser: ").strip()
                if user_query.lower() in ['quit', 'exit', 'bye']:
                    print("Exiting...")
                    break
                query_pdf(pdf_key, user_query)
