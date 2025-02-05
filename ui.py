import os
import tempfile

import fitz
import streamlit as st

from model.intellidocs_rag_final.id_rag_updated import IntellidocsRAG
from utils.constants import ConstantSettings, PathSettings


def load_pdf_pages(file_path):
    """Load PDF pages as images for display."""
    doc = fitz.open(file_path)  # Open PDF from saved file path
    return [page.get_pixmap() for page in doc]


def main():
    st.set_page_config(layout="wide", page_title="Intellidocs")

    # Sidebar for PDF selection and configuration
    st.sidebar.title("Intellidocs: AI powered insights for your documents.")

    # File uploader for PDFs
    uploaded_pdfs = st.sidebar.file_uploader(
        "Upload PDFs for analysis",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_pdfs:
        st.sidebar.subheader("Selected PDFs")
        st.sidebar.write("Displaying top 5 pages from the PDF")

        temp_dir = tempfile.mkdtemp()  # Create a temporary directory
        pdf_paths = []
        pdf_keys = []

        for uploaded_file in uploaded_pdfs:
            # Save uploaded file to temporary location
            temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.sidebar.write(f"📄 {uploaded_file.name}")

            # Display PDF preview in sidebar
            pdf_pages = load_pdf_pages(temp_pdf_path)
            for page_num, page in enumerate(pdf_pages[:5], 1):  # Show first 5 pages
                st.sidebar.image(page.tobytes(), channels='RGB', caption=f'Page {page_num}', use_column_width=True)

            pdf_paths.append(temp_pdf_path)  # Store actual file path
            pdf_keys.append(uploaded_file.name.replace(".pdf", "_key"))

        # Initialize RAG
        rag = IntellidocsRAG(
            pdf_doc_paths=pdf_paths,
            chunk_size=100,
            embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
            chroma_db_dir=PathSettings.CHROMA_DB_PATH
        )

        # Extract text and generate embeddings
        extracted_texts = rag.extract_text_from_documents_fitz()
        chunked_texts = rag.text_chunking(extracted_texts)
        extracted_texts_embeddings = rag.generate_embeddings(chunked_texts)
        rag.store_embeddings(chunked_texts, extracted_texts_embeddings)

        # PDF Key selection for querying
        selected_pdf_key = st.sidebar.selectbox("Select PDF Key for Querying", pdf_keys)

        # Centered Query Component
        st.markdown("<h2 style='text-align: center;'>Query Your Document</h2>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])  # Centering using Streamlit columns
        with col2:
            user_query = st.text_input("Enter your query:", key="query_input", help="Type your question here.")

        # Query results
        if user_query and selected_pdf_key:
            try:
                results = rag.retrieve_top_n(user_query, selected_pdf_key, top_n=5)
                total = ''.join(text for text in results[0]['chunk'])
                # Center the results
                with col2:
                    st.subheader("Query Results")
                    st.write(total)


            except Exception as e:
                st.error(f"Error retrieving results: {e}")


if __name__ == '__main__':
    main()
