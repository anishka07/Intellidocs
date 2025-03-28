import os

from pypdf import PdfReader

from utils.constants import PathSettings


class PDFProcessor:
    def __init__(self, pdf_name: str, pdf_path: str):
        """
        Initializes the PDFProcessor class.

        Args:
            pdf_name (str): The name of the PDF file to process.
            pdf_path (str): The path where the PDF file is stored.
        """
        self.pdf_name = pdf_name
        self.pdf_path = pdf_path
        self.full_path = os.path.join(self.pdf_path, self.pdf_name)
        self.text = ""

    def extract_text(self) -> str:
        """
        Extracts text from the specified PDF file.

        Returns:
            str: Extracted text from the PDF file.
        """
        if not os.path.exists(self.full_path):
            print("Path doesn't exist:", self.full_path)
        else:
            reader = PdfReader(self.full_path)
            pages = reader.pages
            for page in pages:
                self.text += page.extract_text().lower()
        return self.text

    def clean_text(self) -> str:
        """
        Cleans the extracted text by removing unwanted spaces and newlines.

        Returns:
            str: The cleaned text.
        """
        self.text = self.text.replace("\n", " ").strip()
        return self.text


if __name__ == "__main__":
    pdf_processor = PDFProcessor(
        pdf_name="monopoly.pdf", pdf_path=PathSettings.PDF_FILE_PATH
    )
    print(pdf_processor.extract_text())
    print(pdf_processor.clean_text())
