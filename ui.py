import streamlit as st
import os

from model.intellidocs_rag_final.id_chroma_rag import IntellidocsRAG
from utils.constants import PathSettings, ConstantSettings

# Streamlit App Title
st.title("Intellidocs RAG System")
st.markdown("AI powered insights for your documents!")

# Sidebar for user inputs
with st.sidebar:
    st.header("Configuration")
    chunk_size = st.number_input("Chunk Size", min_value=100, max_value=1000, value=ConstantSettings.CHUNK_SIZE)
    embedding_model = st.text_input("Embedding Model", value=ConstantSettings.EMBEDDING_MODEL_NAME)
    chroma_db_dir = st.text_input("Chroma DB Directory", value=PathSettings.CHROMA_DB_PATH)
    collection_name = st.text_input("Collection Name", value=ConstantSettings.CHROMA_DB_COLLECTION)

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    pdf_path = os.path.join(PathSettings.PDF_DIR_PATH, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Initialize the IntellidocsRAG instance
    retriever = IntellidocsRAG(
        pdf_doc_path=pdf_path,
        chunk_size=chunk_size,
        embedding_model=embedding_model,
        chroma_db_dir=chroma_db_dir
    )

    # Extract and chunk text
    st.header("Step 1: Extract and Chunk Text")
    if st.button("Extract Text"):
        with st.spinner("Extracting text from the PDF..."):
            ex_text = retriever.extract_text_from_document_fitz()
            st.session_state.extracted_text = ex_text
            st.success("Text extraction complete!")
            st.text_area("Extracted Text", value=ex_text[:5000] + "...", height=300)

    if st.button("Chunk Text"):
        with st.spinner("Chunking text..."):
            chunks = retriever.text_chunking(extracted_text=st.session_state.extracted_text)
            st.session_state.chunks = chunks
            st.success(f"Text chunking complete! {len(chunks)} chunks created.")
            st.write("Sample Chunks:")
            for i, chunk in enumerate(chunks[:3]):
                st.write(f"Chunk {i + 1}: {chunk[:200]}...")

    # Generate embeddings and store them
    st.header("Step 2: Generate and Store Embeddings")
    if st.button("Generate Embeddings"):
        with st.spinner("Generating embeddings..."):
            embeddings = retriever.generate_embeddings(st.session_state.chunks)
            st.session_state.embeddings = embeddings
            st.success("Embeddings generated successfully!")

    if st.button("Store Embeddings"):
        with st.spinner("Storing embeddings in Chroma DB..."):
            retriever.store_embeddings(st.session_state.chunks, st.session_state.embeddings, collection_name)
            st.success("Embeddings stored in Chroma DB!")

    # Query the Chroma DB
    st.header("Step 3: Query the Document")
    user_query = st.text_input("Enter your query:")
    st.session_state.query = user_query
    if st.button("Search"):
        with st.spinner("Searching for relevant chunks..."):
            results = retriever.retrieve_top_n(st.session_state.query, collection_name, top_n=5)
            if results:
                st.success(f"Found {len(results)} relevant chunks:")
                for i, result in enumerate(results):
                    st.subheader(f"Result {i + 1}")
                    st.write(f"**Chunk:** {result['chunk'][:500]}...")
                    st.write(f"**Score:** {result['score'][0]}")
            else:
                st.warning("No results found for the query.")
else:
    st.warning("Please upload a PDF file to get started.")