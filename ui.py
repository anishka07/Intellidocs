import streamlit as st
import os
from model.intellidocs_rag_final.chunk_processor import tp_main
from model.intellidocs_rag_final.embedding_process import embedding_process_main
from model.intellidocs_rag_final.intellidocs_rag_constants import sent_tokenizer_model_name
from model.intellidocs_rag_final.pdf_loader import pdf_loader_main
from model.intellidocs_rag_final.retrieval_process import retriever_main
from utils.constants import PathSettings

st.title("IntelliDocs: PDF Query System")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_path = os.path.join(PathSettings.PDF_DIR_PATH, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"PDF {uploaded_file.name} uploaded successfully!")


    @st.cache_data
    def process_pdf(pdf_name):
        extracted_text = pdf_loader_main(path=PathSettings.PDF_DIR_PATH, pdf_name=pdf_name)
        filtered_chunks = tp_main(pgs_texts=extracted_text, min_token_len=30)
        save_csv_name = f"{pdf_name.split('.')[0]}_embeddings.csv"
        if embedding_process_main(
                embedding_model_name=sent_tokenizer_model_name,
                pages_and_chunks=filtered_chunks,
                project_dir=PathSettings.PROJECT_DIR_PATH,
                csv_name=save_csv_name,
                save_dir="CSV_db",
                dev="cpu"
        ):
            st.success("Embeddings created successfully.")
        return save_csv_name


    csv_name = process_pdf(uploaded_file.name)

    st.write("You can now query the document.")
    user_query = st.text_input("Enter your query:")

    if user_query:
        results = retriever_main(
            embeddings_df_path=os.path.join(PathSettings.CSV_DB_DIR_PATH, csv_name),
            user_query=user_query
        )

        if results:
            st.write("Top results:")
            for result in results:
                st.write(f"**Score:** {result['score']:.4f}")
                st.write(f"**Page Number:** {result['page_number']}")
                st.write(f"**Text Chunk:** {result['text']}")
                st.write("\n---\n")
        else:
            st.write("No results found.")

if st.button("Clear Cache and Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.write("Cache and data cleared.")
