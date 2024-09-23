import re

import pandas as pd
from tqdm.auto import tqdm

from model.intellidocs_rag_final.intellidocs_rag_constants import id_rag_pdf_path, id_pdf_name
from model.intellidocs_rag_final.pdf_loader import pdf_loader_main


class ChunkProcessor:
    def __init__(self, pages_and_texts: list[dict], min_token_length: int):
        self.pages_and_texts = pages_and_texts
        self.min_token_length = min_token_length
        self.data_frame = pd.DataFrame(self.process_chunks())

    def process_chunks(self):
        pages_and_chunks = []
        for item in tqdm(self.pages_and_texts):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {"page_number": item["page_number"]}

                # Join the sentences together into a paragraph-like structure
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)  # Fix ".A" to ". A"

                chunk_dict["sentence_chunk"] = joined_sentence_chunk

                # Get stats about the chunk
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split(" "))
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters

                pages_and_chunks.append(chunk_dict)
        return pages_and_chunks

    def filter_chunks_by_token_length(self):
        pages_and_chunks_over_min_token_len = self.data_frame[
            self.data_frame["chunk_token_count"] > self.min_token_length].to_dict(
            orient="records")
        return pages_and_chunks_over_min_token_len

    def inspect_short_chunks(self, num_samples: int = 5):
        """Inspect a sample of short chunks for debugging"""
        min_token_length = self.min_token_length
        sample_rows = self.data_frame[self.data_frame["chunk_token_count"] <= min_token_length].sample(
            num_samples).iterrows()
        for row in sample_rows:
            print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')


def tp_main(pgs_texts: list[dict], min_token_len: int):
    cp = ChunkProcessor(pgs_texts, min_token_length=min_token_len)
    filtered_sentence_chunks = cp.filter_chunks_by_token_length()
    return filtered_sentence_chunks


if __name__ == '__main__':
    extracted_text_dict = pdf_loader_main(path=id_rag_pdf_path, pdf_name=id_pdf_name)

    chunk_processor = ChunkProcessor(extracted_text_dict, min_token_length=30)
    df = chunk_processor.process_chunks()

    filtered_chunks = chunk_processor.filter_chunks_by_token_length()
    print(filtered_chunks[:2])

    chunk_processor.inspect_short_chunks(num_samples=5)
