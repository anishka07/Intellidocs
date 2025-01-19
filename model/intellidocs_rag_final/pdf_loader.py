import logging
import os

import fitz
import requests
from pypdf import PdfReader
from spacy.lang.en import English
from tqdm.auto import tqdm

from model.intellidocs_rag_final.intellidocs_rag_constants import id_rag_pdf_path
from utils.constants import PathSettings


def download_and_save_pdf(pdf_url: str, saved_pdf_name: str):
    """

    :param pdf_url: url of the pdf you want to download
    :param saved_pdf_name: name of the pdf you want to save
    :return: saved or not, response
    """
    save_pdf_path = os.path.join(PathSettings.PDF_DIR_PATH, saved_pdf_name)
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(save_pdf_path, "wb") as f:
            f.write(response.content)
            logging.info(f"Saved PDF to {save_pdf_path}.")
    else:
        logging.error(f"Failed to download PDF from {pdf_url}.")


class PdfLoader:

    def __init__(self, pdf_path: str, pdf_url: str = None, save_pdf_name: str = None) -> None:
        self.pdf_path = pdf_path
        self.pdf_url = pdf_url
        if self.pdf_url is not None and self.pdf_path is None:
            download_and_save_pdf(pdf_url=self.pdf_url, saved_pdf_name=save_pdf_name)

    def open_and_read_pdf(self) -> str:
        all_text = ""
        reader = PdfReader(self.pdf_path)
        for page in tqdm(reader.pages):
            text = page.extract_text()
            all_text += text
        return all_text

    def fitz_open_and_load_pdf(self):
        docs = fitz.open(self.pdf_path)
        page_and_text = []
        for page_number, page in tqdm(enumerate(docs)):
            text = page.get_text()
            text = text.replace("\n", "").strip()
            page_and_text.append({"page_number": page_number,  # - 41 for the health pdf
                                  "page_char_count": len(text),
                                  "page_word_count": len(text.split(" ")),
                                  "page_sentence_count_raw": len(text.split(". ")),
                                  "page_token_count": len(text) / 4,
                                  "text": text})
        return page_and_text

    def add_tokenized_sentences(self):
        num_chunks = 10

        def split_list(input_list: list, slice_size: int):
            return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

        pages_and_texts_dict = self.fitz_open_and_load_pdf()
        nlp = English()
        nlp.add_pipe('sentencizer')
        for item in tqdm(pages_and_texts_dict):
            item["sentences"] = list(nlp(item['text']).sents)
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            item["page_sentence_count_spacy"] = len(item["sentences"])
            item["sentence_chunks"] = split_list(item["sentences"], num_chunks)
            item["num_chunks"] = len(item["sentence_chunks"])
        return pages_and_texts_dict


def pdf_loader_main(path: str, pdf_name: str):
    pdf_loader = PdfLoader(pdf_path=os.path.join(path, pdf_name))
    extracted_dict = pdf_loader.add_tokenized_sentences()
    return extracted_dict


if __name__ == '__main__':
    pages_and_texts = pdf_loader_main(path=id_rag_pdf_path, pdf_name='frog.pdf')
    print(pages_and_texts[0:5])
