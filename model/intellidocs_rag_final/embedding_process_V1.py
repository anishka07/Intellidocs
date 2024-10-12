from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingProcessorV1:

    def __init__(self, embedding_model_name: str, pages_and_chunks: list[dict], project_dir: str, collection_name: str):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_model.tokenizer.clean_up_tokenization_spaces = True
        self.pages_and_chunks = pages_and_chunks
        self.project_dir = project_dir
        self.collection_name = collection_name
        # Connect to milvus
        connections.connect("default", host="localhost", port="19530")

    def move_model_to_device(self, device: str = "cpu"):
        self.embedding_model.to(device)

    def encode_chunk(self, chunk: str):
        return self.embedding_model.encode(chunk)

    def create_collection(self):
        if utility.has_collection(self.collection_name):
            return Collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="sentence_chunk", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768)  # Adjust dimension as needed
        ]
        schema = CollectionSchema(fields, "PDF Chunks Collection")
        collection = Collection(self.collection_name, schema)

        # Create an IVF_FLAT index for the embeddings field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("embeddings", index_params)
        return collection

    def add_embeddings_to_chunks(self):
        collection = self.create_collection()
        entities = []
        for item in tqdm(self.pages_and_chunks):
            embedding = self.encode_chunk(item["sentence_chunk"])
            entities.append([
                item["page_number"],
                item["sentence_chunk"],
                embedding.tolist()
            ])

        collection.insert(entities)
        collection.flush()
        return self.collection_name


def embedding_process_main(embedding_model_name: str, pages_and_chunks: list[dict], project_dir: str,
                           collection_name: str, dev: str):
    ep = EmbeddingProcessorV1(embedding_model_name, pages_and_chunks, project_dir, collection_name)
    ep.move_model_to_device(device=dev)
    return ep.add_embeddings_to_chunks()
