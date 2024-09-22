import os
import re

import faiss
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
# from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from utils.constants import ConstantSettings

load_dotenv()
GA_API_KEY = os.getenv("GOOGLE_GEMINI_API_TOKEN")
genai.configure(api_key=GA_API_KEY)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


model = SentenceTransformer(ConstantSettings.EMBEDDING_MODEL_NAME)


def chunk_text(text, chunk_size: int = 1000, chunk_overlap: int = 20):
    def split_into_sentences(text):
        text = text.replace("\n", " ")
        return re.split(r'(?<=[.!?])\s+', text)

    def get_overlap_words(text):
        words = text.split()
        return words[-chunk_overlap:] if len(words) > chunk_overlap else words

    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_chunk_size + sentence_length > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap_words = get_overlap_words(" ".join(current_chunk))
                current_chunk = overlap_words
                current_chunk_size = len(" ".join(overlap_words))

        current_chunk.append(sentence)
        current_chunk_size += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# def get_vector_store(text_chunks: list[str]):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embeddings)
#     vector_store.save_local("faiss_index")
def get_vector_store(text_chunks: list[str]):
    # Generate embeddings
    embeddings = model.encode(text_chunks)

    # Create a FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add vectors to the index
    index.add(embeddings.astype('float32'))

    # Save the index
    faiss.write_index(index, "faiss_index")


def user_input(user_question):
    # Load the saved index
    index = faiss.read_index("faiss_index")

    # Encode the user question
    question_embedding = model.encode([user_question])

    # Search the index
    k = 4  # number of nearest neighbors to retrieve
    distances, indices = index.search(question_embedding.astype('float32'), k)

    # Retrieve the corresponding text chunks
    relevant_chunks = [text_chunks[i] for i in indices[0]]

    # Use these relevant chunks in your conversational chain
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": relevant_chunks, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=.3)
    prompt = PromptTemplate(prompt=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#
#     new_db = FAISS.load_local("faiss_index", embeddings)
#     docs = new_db.similarity_search(user_question)
#
#     chain = get_conversational_chain()
#
#     response = chain(
#         {"input_documents": docs, "question": user_question}
#         , return_only_outputs=True)
#
#     print(response)
#     st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = chunk_text(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == '__main__':
    main()
