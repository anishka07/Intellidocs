import torch
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

from model.intellidocs_rag_v3.intellidocs_rag_constants import sent_tokenizer_model_name


class Retriever:
    def __init__(self, collection_name: str, model_name: str = sent_tokenizer_model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(model_name_or_path=model_name, device=self.device)
        self.collection_name = collection_name

        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def retrieve_relevant_resources(self, query: str, n_resources_to_return: int = 5):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embeddings",
            param=search_params,
            limit=n_resources_to_return,
            output_fields=["sentence_chunk", "page_number"]
        )
        return results[0]

    def print_top_results_and_scores(self, query: str, n_resources_to_return: int = 5):
        results = self.retrieve_relevant_resources(query=query, n_resources_to_return=n_resources_to_return)

        print(f"Query: {query}\n")
        print("Results:")
        for hit in results:
            print(f"Score: {hit.score:.4f}")
            self.print_wrapped(hit.entity.get('sentence_chunk'))
            print(f"Page number: {hit.entity.get('page_number')}")
            print("\n")

    @staticmethod
    def print_wrapped(text: str, width: int = 100):
        import textwrap
        print(textwrap.fill(text, width=width))


def retriever_main(collection_name: str, user_query: str):
    retriever = Retriever(collection_name)
    results = retriever.retrieve_relevant_resources(user_query)

    return [
        {
            "score": hit.score,
            "text": hit.entity.get('sentence_chunk'),
            "page_number": hit.entity.get('page_number')
        }
        for hit in results
    ]
