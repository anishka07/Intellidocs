import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import util, SentenceTransformer

from model.intellidocs_rag_v3.intellidocs_rag_constants import sent_tokenizer_model_name
from utils.constants import PathSettings


class Retriever:

    def __init__(self, embeddings_df_path: str, model_name: str = sent_tokenizer_model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(model_name_or_path=model_name, device=self.device)
        self.embeddings_df = self._load_embeddings(embeddings_df_path)
        self.pages_and_chunks = self.embeddings_df.to_dict(orient="records")
        self.embeddings = self._prepare_embeddings()

    def _load_embeddings(self, embeddings_df_path: str) -> pd.DataFrame:
        df = pd.read_csv(embeddings_df_path)
        df["embeddings"] = df["embeddings"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        self.embedding_model_name = df['model_name'].iloc[0]
        return df

    def _prepare_embeddings(self) -> torch.Tensor:
        return torch.tensor(np.array(self.embeddings_df["embeddings"].tolist()), dtype=torch.float32).to(self.device)

    def retrieve_relevant_resources(self,
                                    query: str,
                                    n_resources_to_return: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds a query with model and returns top k scores and indices from embeddings.
        """
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        dot_scores = util.dot_score(query_embedding, self.embeddings)[0]
        scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

        return scores, indices

    def print_top_results_and_scores(self,
                                     query: str,
                                     n_resources_to_return: int = 5):
        """
        Takes a query, retrieves most relevant resources and prints them out in descending order.
        """
        scores, indices = self.retrieve_relevant_resources(query=query, n_resources_to_return=n_resources_to_return)

        print(f"Query: {query}\n")
        print("Results:")
        for score, index in zip(scores, indices):
            print(f"Score: {score:.4f}")
            self.print_wrapped(self.pages_and_chunks[index]["sentence_chunk"])
            print(f"Page number: {self.pages_and_chunks[index]['page_number']}")
            print("\n")

    @staticmethod
    def print_wrapped(text: str, width: int = 100):
        """Utility method to print wrapped text"""
        import textwrap
        print(textwrap.fill(text, width=width))


def retriever_main(embeddings_df_path: str, user_query: str):
    retriever = Retriever(embeddings_df_path)
    scores, indices = retriever.retrieve_relevant_resources(user_query)

    results = []
    for score, index in zip(scores, indices):
        result = {
            "score": score.item(),
            "text": retriever.pages_and_chunks[index]["sentence_chunk"],
            "page_number": retriever.pages_and_chunks[index]["page_number"]
        }
        results.append(result)
    return results


if __name__ == "__main__":
    retriever = Retriever(os.path.join(PathSettings.CSV_DB_DIR_PATH, "test1.csv"))
    query = "macronutrients"
    retriever.print_top_results_and_scores(query)
