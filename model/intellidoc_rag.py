import os
import shutil
from typing import List

from langchain_community.vectorstores import Chroma
from nltk.tokenize import sent_tokenize
from pymupdf import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from utils.constants import PathSettings, ConstantSettings

# client = chromadb.PersistentClient(path=PathSettings.CHROMA_DB_PATH)
# collection = client.create_collection(name="pdf_embeddings1")
model = SentenceTransformer(ConstantSettings.EMBEDDING_MODEL_NAME)
CHROMA_PATH = "chroma"


def text_extractor(pdf_name: str) -> str:
    """
    Extracts text from pdf file
    Args:
        pdf_name: name of the pdf you want to extract texts from

    Returns:
        str: extracted text
    """
    all_text = ""
    pdf = os.path.join(PathSettings.PDF_FILE_PATH, pdf_name)
    if not os.path.exists(pdf):
        print("Path doesn't exist: ", PathSettings.PDF_FILE_PATH)
    else:
        reader = PdfReader(pdf)
        pages = reader.pages
        for page in pages:
            all_text += page.extract_text()
            all_text = all_text.lower()
    return all_text


def text_cleaner(text: str) -> str:
    """
    Removes white space from the extracted text
    Args:
        text: the text you want to clean

    Returns:
        str: the cleaned text
    """
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text


def sentence_tokenizer(all_texts: str) -> List[str]:
    """
    Tokenizes the paragraph based on sentences ("./full stop").
    Args:
        all_texts: the paragraph of text you want to tokenize

    Returns:
        list[str]: list of sentences
    """
    tokenized_sentences = sent_tokenize(all_texts)
    return tokenized_sentences


def split_text_to_chunks(input_list: list,
                         slice_size: int = 10) -> list[list[str]]:
    """
    Splits the tokenized list into chunks, default value = 10.
    Args:
        input_list: list of sentences
        slice_size: number of chunks to split the list into

    Returns:
        chunked list selected slice_size
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


def embed_chunks(chunked_text: List[List[str]]) -> list[List[float]]:
    """
    Embeds each chunk of text using a pre-trained sentence transformer model
    Args:
        chunked_text: list of text chunks

    Returns:
        List of embeddings for each chunk
    """
    embeddings = []
    for chunk in chunked_text:
        combined_text = ' '.join(chunk)
        # Sentence-Transformers returns NumPy arrays by default
        chunk_embeddings = model.encode(combined_text)
        # Convert the NumPy array to a list
        embeddings.append(chunk_embeddings.tolist())
    return embeddings


def save_to_chroma(chunks: list[list[str]]):
    """
  Save the given list of Document objects to a Chroma database.
  Args:
  chunks (list[Document]): List of Document objects representing text chunks to save.
  Returns:
  None
  """

    # Clear out the existing database directory if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new Chroma database from the documents using OpenAI embeddings
    db = Chroma.from_documents(
        chunks,
        SentenceTransformer(ConstantSettings.EMBEDDING_MODEL_NAME),
        persist_directory=CHROMA_PATH
    )

    # Persist the database to disk
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == '__main__':
    texts = text_extractor("monopoly.pdf")
    clean_text = text_cleaner(texts)
    sentences = sentence_tokenizer(clean_text)
    save_to_chroma(sentences)
