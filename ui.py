import streamlit as st
import os

from model.intellidocs_main import intelli_docs_main
from utils.constants import PathSettings

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
            response = intelli_docs_main(
                user_query=user_query,
                save_csv_name=save_csv_name,
                save_csv_dir=PathSettings.CSV_DB_DIR_PATH,
                rag_device="cpu"
            )

        if response:
            # Append the query and response to the chat log
            st.session_state.chat_log.append({"query": user_query, "response": response})
        else:
            st.session_state.chat_log.append({"query": user_query, "response": "No results found."})

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
