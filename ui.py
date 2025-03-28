import base64
import logging
import os
import tempfile

import streamlit as st

from src.llms.gemini_response import gemini_response
from src.model.intellidocs_rag.intellidocs_main import IntellidocsRAG
from utils.constants import ConstantSettings, PathSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG
rag = IntellidocsRAG(
    chunk_size=ConstantSettings.CHUNK_SIZE,
    embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
    chroma_db_dir=PathSettings.CHROMA_DB_PATH,
)


def load_pdf_as_base64(file_path):
    """Convert PDF file to base64 string."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    return base64_pdf


def main():
    st.set_page_config(layout="wide", page_title="Intellidocs")

    # Sidebar for PDF selection and configuration
    st.sidebar.title("Intellidocs: AI powered insights for your documents.")

    # File uploader for PDFs
    uploaded_pdfs = st.sidebar.file_uploader(
        "Upload PDFs for analysis", type=["pdf"], accept_multiple_files=True
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

            # Convert PDF to base64
            base64_pdf = load_pdf_as_base64(temp_pdf_path)

            # Create PDF viewer in tab
            with pdf_tab:
                # PDF viewer with custom HTML and CSS
                pdf_viewer_html = f"""
                    <div style="display: flex; justify-content: center; width: 100%;">
                        <iframe
                            src="data:application/pdf;base64,{base64_pdf}"
                            width="100%"
                            height="600px"
                            style="border: 1px solid #ccc; border-radius: 5px;"
                            type="application/pdf"
                        >
                        </iframe>
                    </div>
                """
                st.markdown(pdf_viewer_html, unsafe_allow_html=True)

            pdf_paths.append(temp_pdf_path)
            pdf_keys.append(uploaded_file.name.replace(".pdf", "_key"))
            rag.process(pdf_paths)

        # PDF selection for querying
        selected_pdf_key = st.sidebar.selectbox(
            "Select PDF for Querying",
            pdf_keys,
            format_func=lambda x: x.replace("_key", ".pdf"),
        )

        # Main query interface
        st.markdown(
            "<h2 style='text-align: center;'>Query Your Document</h2>",
            unsafe_allow_html=True,
        )

        # Center the query input
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            user_query = st.text_input(
                "Enter your query:", key="query_input", help="Type your question here."
            )

            # Process query and display results
            if user_query and selected_pdf_key:
                try:
                    st.subheader(f"Query Results for {selected_pdf_key}: ")
                    results = rag.retrieve_top_n(
                        user_query, selected_pdf_key, top_n=5
                    )
                    st.subheader("Response from LLM: ")
                    top_texts = []
                    for item in results:
                        chunk = item["chunk"].strip()
                        score = item["score"]
                        top_texts.append({"text": chunk, "score": score})
                    llm_response = gemini_response(
                        user_query=user_query,
                        context=top_texts,
                    )
                    st.write(llm_response)
                    if not results:
                        st.write("No results found.")
                    else:
                        st.subheader("Retrieved Chunks (ordered by relevance): ")
                        st.write("-" * 80)
                        for i, item in enumerate(top_texts, 1):
                            st.write(
                                f"\nChunk {i} (Similarity Score: {item['score']:.4f})"
                            )
                            st.write(item["text"])
                            st.write("-" * 80)

                except Exception as e:
                    st.error(f"Error retrieving results: {str(e)}")


if __name__ == "__main__":
    main()
