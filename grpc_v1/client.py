import logging
import sys
from typing import List, Iterator

import grpc

from intellidocspb.v1 import intellidocs_pb2 as intellidocs_pb2
from intellidocspb.v1 import intellidocs_pb2_grpc as intellidocs_pb2_grpc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntelliDocsClient:
    """
    gRPC client for IntelliDocs
    """
    def __init__(self, server_address: str = "localhost:50051"):
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self.document_ids = {}

    def connect(self):
        """
        Connect to RPC server
        """
        logger.info("Connecting to server at " + self.server_address)
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = intellidocs_pb2_grpc.IntelliDocsServiceStub(self.channel)
        logger.info("Connection established.")

    def close(self):
        """
        Close connection to RPC server
        """
        if self.channel:
            self.channel.close()
            logger.info("Connection closed.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _generate_requests(
            self,
            document_id: str,
            queries: List[str]
    ) -> Iterator[intellidocs_pb2.ChatWithDocumentRequest]:
        for query in queries:
            yield intellidocs_pb2.ChatWithDocumentRequest(
                document_id=document_id,
                user_query=query
            )

    def chat_with_document(self, document_id: str, queries: List[str]):
        if not self.stub:
            raise RuntimeError("Stub is not connected. Call connect() first.")
        logger.info(f"Sending {len(queries)} queries for document {document_id}")
        try:
            request_iterator = self._generate_requests(document_id, queries)

            responses = []
            for response in self.stub.ChatWithDocument(request_iterator):
                responses.append(response.query_response)
            return responses
        except grpc.RpcError as e:
            status_code = e.code()
            status_details = e.details()
            logger.error(f"Error {status_code} - {status_details}")
            raise

    def interactive_chat(self, document_id: str):
        if not self.stub:
            raise RuntimeError("Stub is not connected. Call connect() first.")

        logger.info(f"Starting interactive chat with document {document_id}")
        print(f"Starting chat with document {document_id}")
        print("Type 'exit' or 'quit' to end the session.")
        try:
            while True:
                query = input("\nYour query: ")
                if query.lower() in ("exit", "quit"):
                    break

                request_iterator = self._generate_requests(document_id, [query])

                print("\nResponse:")
                for response in self.stub.ChatWithDocument(request_iterator):
                    for i, chunk in enumerate(response.query_response):
                        print(f"[{i + 1}] {chunk}")
        except grpc.RpcError as e:
            logger.error(f"Error {e.code()} - {e.details()}")
            print(f"Error: {e.details()}")
        except KeyboardInterrupt:
            print("\nGoodbye.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive gRPC client")
    parser.add_argument("--server", type=str, default="localhost:50051",
                        help="Server address in format host:port")
    parser.add_argument("--document-id", type=str, required=True,
                        help="ID of the document to interact with")
    args = parser.parse_args()

    with IntelliDocsClient(args.server) as client:
        try:
            client.interactive_chat(args.document_id)
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            sys.exit(1)


if __name__ == '__main__':
    main()
