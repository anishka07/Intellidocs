from typing import List

from nltk import sent_tokenize

from model.document_processor import PDFProcessor
from utils.constants import PathSettings


class TextProcessor:

    def __init__(self, text: str, slice_size: int):
        self.slice_size = slice_size
        self.text = text

    def sentence_tokenizer(self) -> List[str]:
        """
        Tokenizes the text into sentences using full stops.

        Returns:
            list[str]: A list of sentences.
        """
        return sent_tokenize(self.text)

    def split_text_to_chunks(self) -> List[List[str]]:
        """
        Splits tokenized sentences into chunks of a specified size.

        Returns:
            list[list[str]]: A list of chunks, each containing a slice of the sentences.
        """
        sentences = self.sentence_tokenizer()
        return [sentences[i:i + self.slice_size] for i in range(0, len(sentences), self.slice_size)]


if __name__ == '__main__':
    pdf_processor = PDFProcessor(pdf_name='monopoly.pdf', pdf_path=PathSettings.PDF_FILE_PATH)
    extracted_text = pdf_processor.extract_text()
    cleaned_text = pdf_processor.clean_text()

    text_processor = TextProcessor(cleaned_text, slice_size=5)
    chunks = text_processor.split_text_to_chunks()
