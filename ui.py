import logging
import os

import streamlit as st

from model.intellidocs_main import intelli_docs_main
from model.llms.gemini_response import gemini_response
from utils.constants import PathSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("IntelliDocs: AI powered insights for your documents.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if uploaded_file is not None:
    pdf_path = os.path.join(PathSettings.PDF_DIR_PATH, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"PDF {uploaded_file.name} uploaded successfully!")

    user_query = st.text_input("Enter your query:", key="user_query")

    if user_query:
        save_csv_name = f"{uploaded_file.name.split('.')[0]}_embeddings.csv"
        save_csv_path = os.path.join(PathSettings.CSV_DB_DIR_PATH, save_csv_name)

        if os.path.exists(save_csv_path):
            st.write(f"Using existing CSV file: {save_csv_name}")
        else:
            st.write("CSV file not found. Generating embeddings...")

        with st.spinner("Processing your query and generating response..."):
            # Step 1: Generate RAG response
            rag_response = intelli_docs_main(
                user_query=user_query,
                save_csv_name=save_csv_name,
                save_csv_dir=PathSettings.CSV_DB_DIR_PATH,
                rag_device="cpu"
            )

            # Step 2: Generate Gemini response based on RAG response
            if rag_response:
                llm_response = gemini_response(user_query, context=rag_response)
                response = f"{rag_response}\n\nGemini's insights:\n{llm_response}"
            else:
                response = "No results found. Please try a different query."

        # Append the query and response to the chat log
        st.session_state.chat_log.append({"query": user_query, "response": response})

# Display chat log
if st.session_state.chat_log:
    st.write("### Chat Log")
    for entry in st.session_state.chat_log:
        st.write(f"**User:** {entry['query']}")
        st.write(f"**IntelliDocs:** {entry['response']}")
        st.write("---")

if st.button("Clear Chat Log"):
    st.session_state.chat_log = []
    st.write("Chat log cleared.")

if st.button("Clear Cache and Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.write("Cache and data cleared.")
