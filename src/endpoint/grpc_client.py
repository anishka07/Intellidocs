import os 
import grpc 

from src.intellidocspb import intellidocs_pb2
from src.intellidocspb import intellidocs_pb2_grpc
from utils.constants import PathSettings


def run(user_query, dco, document_key):
    channel = grpc.insecure_channel('localhost:50051')
    stub = intellidocs_pb2_grpc.IntelliDocsServiceStub(channel=channel)

    response = stub.ProcessDocuments(
        intellidocs_pb2.ProcessDocumentRequest(
            document_paths=[os.path.join(PathSettings.PDF_DIR_PATH, dco)]
        )
    )
    print(f"Processed documents: {response.message}")

    retrieve_response = stub.RetrieveTopN(
        intellidocs_pb2.RetrieveTopNRequest(
            user_query=user_query,
            doc_key=document_key,
            top_n=3
        )
    )
    print("Top chunks:")
    for chunk in retrieve_response.chunks:
        print(f"- {chunk.chunks} (score: {chunk.score})")


if __name__ == "__main__":
    dco = str(input("Enter your document path: "))
    query = str(input("enter your query:"))
    document_key = str(input("enter document key:"))
    run(user_query=query, dco=dco, document_key=document_key)

