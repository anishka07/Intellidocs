import os
import tempfile
import fitz
import streamlit as st
import logging

from model.intellidocs_rag_final.id_rag_updated import IntellidocsRAG
from utils.constants import ConstantSettings, PathSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pdf_pages(file_path):
    """Load PDF pages as images for display."""
    doc = fitz.open(file_path)
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

        # Create tabs for each PDF
        pdf_tabs = st.sidebar.tabs([pdf.name for pdf in uploaded_pdfs])

        temp_dir = tempfile.mkdtemp()
        pdf_paths = []
        pdf_keys = []

        # Process each PDF
        for idx, (uploaded_file, pdf_tab) in enumerate(zip(uploaded_pdfs, pdf_tabs)):
            # Save uploaded file
            temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load PDF pages
            pdf_pages = load_pdf_pages(temp_pdf_path)

            # Create scrollable container for PDF preview
            with pdf_tab:
                preview_container = st.container()
                with preview_container:
                    st.write("PDF Preview (scrollable)")
                    preview_area = st.empty()
                    selected_page = st.slider(
                        "Select page",
                        1,
                        len(pdf_pages),
                        1,
                        key=f"page_slider_{idx}"
                    )
                    # Display selected page
                    preview_area.image(
                        pdf_pages[selected_page - 1].tobytes(),
                        caption=f'Page {selected_page}',
                        use_column_width=True
                    )

            pdf_paths.append(temp_pdf_path)
            pdf_keys.append(uploaded_file.name.replace(".pdf", "_key"))

        # Initialize RAG
        rag = IntellidocsRAG(
            pdf_doc_paths=pdf_paths,
            chunk_size=400,
            embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
            chroma_db_dir=PathSettings.CHROMA_DB_PATH
        )

        # Process documents
        extracted_texts = rag.extract_text_from_documents_fitz()
        chunked_texts = rag.text_chunking(extracted_texts)
        extracted_texts_embeddings = rag.generate_embeddings(chunked_texts)
        rag.store_embeddings(chunked_texts, extracted_texts_embeddings)

        # PDF selection for querying
        selected_pdf_key = st.sidebar.selectbox(
            "Select PDF for Querying",
            pdf_keys,
            format_func=lambda x: x.replace("_key", ".pdf")
        )

        # Main query interface
        st.markdown("<h2 style='text-align: center;'>Query Your Document</h2>", unsafe_allow_html=True)

        # Center the query input
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            user_query = st.text_input(
                "Enter your query:",
                key="query_input",
                help="Type your question here."
            )

            # Process query and display results
            if user_query and selected_pdf_key:
                try:
                    results = rag.retrieve_top_n(user_query, selected_pdf_key, top_n=5)
                    print(results)
                    st.write(results)
                except Exception as e:
                    st.error(f"Error retrieving results: {str(e)}")


if __name__ == '__main__':
    main()
