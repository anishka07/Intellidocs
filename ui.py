# import os
# import time
#
# import streamlit as st
#
# from model.intellidocs_rag_final.id_chroma_rag import IntellidocsRAG
# from model.llms.gemini_response import gemini_response
# from utils.constants import PathSettings, ConstantSettings
#
#
# def initialize_rag_system(pdf_path: str):
#     """Initialize the RAG system with the given PDF."""
#     return IntellidocsRAG(
#         pdf_doc_path=pdf_path,
#         chunk_size=ConstantSettings.CHUNK_SIZE,
#         embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,  # This model produces 384-dimensional embeddings
#         chroma_db_dir=os.path.join(PathSettings.CHROMA_DB_PATH, "intellidocs_db")
#     )
#
#
# def process_pdf(rag_system: IntellidocsRAG):
#     """Process the PDF and store embeddings."""
#     # Extract text
#     extracted_text = rag_system.extract_text_from_document_fitz()
#
#     # Create chunks
#     text_chunks = rag_system.text_chunking(extracted_text)
#
#     # Generate embeddings
#     embeddings = rag_system.generate_embeddings(text_chunks)
#
#     # Store in ChromaDB with a unique collection name based on PDF name and timestamp
#     collection_name = f"{os.path.basename(rag_system.pdf_path)}_{int(time.time())}"
#     rag_system.store_embeddings(text_chunks, embeddings, collection_name)
#
#     return collection_name
#
#
# def main():
#     st.set_page_config(page_title="IntelliDocs RAG System", layout="wide")
#
#     st.title("IntelliDocs RAG System")
#     st.write("AI generated insights for your documents.")
#
#     # Session state initialization
#     if 'processed_pdf' not in st.session_state:
#         st.session_state.processed_pdf = False
#     if 'collection_name' not in st.session_state:
#         st.session_state.collection_name = None
#     if 'rag_system' not in st.session_state:
#         st.session_state.rag_system = None
#
#     # File uploader
#     uploaded_pdf = st.file_uploader("Upload your PDF file", type=["pdf"])
#
#     # Process PDF
#     if uploaded_pdf is not None:
#         save_pdf_path = os.path.join(PathSettings.PDF_DIR_PATH, uploaded_pdf.name)
#
#         # Only process if it's a new PDF or not processed yet
#         if not st.session_state.processed_pdf:
#             with st.spinner("Processing PDF..."):
#                 # Save the uploaded file
#                 with open(save_pdf_path, "wb") as f:
#                     f.write(uploaded_pdf.read())
#
#                 # Initialize RAG system
#                 st.session_state.rag_system = initialize_rag_system(save_pdf_path)
#
#                 # Process the PDF
#                 try:
#                     st.session_state.collection_name = process_pdf(st.session_state.rag_system)
#                     st.session_state.processed_pdf = True
#                     st.success("PDF processed successfully!")
#                 except Exception as e:
#                     st.error(f"Error processing PDF: {str(e)}")
#                     return
#
#         # Query interface
#         with st.container():
#             col1, col2 = st.columns([3, 1])
#
#             with col1:
#                 user_query = st.text_area("Enter your query:", height=100)
#
#             with col2:
#                 top_k = st.number_input("Number of relevant chunks:", min_value=1, max_value=10, value=3)
#                 submit_button = st.button("Submit Query")
#
#         if user_query and submit_button:
#             with st.spinner("Processing query..."):
#                 try:
#                     # Retrieve relevant chunks
#                     results = st.session_state.rag_system.retrieve_top_n(
#                         user_query=user_query,
#                         chroma_collection_name=st.session_state.collection_name,
#                         top_n=top_k
#                     )
#
#                     if results:
#                         # Display relevant chunks
#                         st.subheader("Relevant Document Chunks:")
#                         for i, result in enumerate(results, 1):
#                             with st.expander(f"Chunk {i} (Distance: {result['score']:.3f})"):
#                                 st.write(result['chunk'])
#
#                         # Combine chunks for context
#                         all_context = " ".join([r['chunk'] for r in results])
#
#                         # Get LLM response
#                         with st.spinner("Generating response..."):
#                             llm_response = gemini_response(
#                                 user_query=user_query,
#                                 context=all_context,
#                             )
#
#                         if llm_response:
#                             st.subheader("AI Response:")
#                             st.write(llm_response)
#                         else:
#                             st.warning("Could not generate a response from the LLM.")
#                     else:
#                         st.warning("No relevant content found in the document.")
#
#                 except Exception as e:
#                     st.error(f"Error processing query: {str(e)}")
#
#     # Clear session state button
#     if st.session_state.processed_pdf:
#         if st.button("Process New PDF"):
#             st.session_state.processed_pdf = False
#             st.session_state.collection_name = None
#             st.session_state.rag_system = None
#             st.experimental_rerun()
#
#
# if __name__ == "__main__":
#     main()

import streamlit as st
import os

from model.intellidocs_rag_final.id_chroma_rag import IntellidocsRAG
from utils.constants import PathSettings, ConstantSettings

# Streamlit App Title
st.title("Intellidocs RAG System")
st.markdown("Upload a PDF, extract text, generate embeddings, and query the document using Chroma DB.")

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