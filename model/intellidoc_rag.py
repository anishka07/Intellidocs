import os
from typing import List

from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from torch import Tensor

from utils.constants import PathSettings, ConstantSettings

model = SentenceTransformer(ConstantSettings.EMBEDDING_MODEL_NAME)


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


def embed_chunks(chunked_text: List[List[str]]) -> list[Tensor]:
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
        chunk_embeddings = model.encode(combined_text)
        embeddings.append(chunk_embeddings)
    return embeddings


if __name__ == '__main__':
    texts = text_extractor("monopoly.pdf")
    clean_text = text_cleaner(texts)
    sentences = sentence_tokenizer(clean_text)
    split_sentences = split_text_to_chunks(sentences)
    embed = embed_chunks(split_sentences)
    print(embed)
