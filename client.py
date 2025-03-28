import argparse
import grpc

from src.intellidocspb.v2 import intellidocs_v2_pb2, intellidocs_v2_pb2_grpc


def process_pdfs(stub, pdf_paths):
    process_response = stub.ProcessPDFs(intellidocs_v2_pb2.ProcessPDFsRequest(pdf_paths=pdf_paths))
    print("PDF Keys: ")
    for pdf_path, pdf_key in process_response.pdf_keys.items():
        print(f"{pdf_path}: {pdf_key}")
    return process_response.pdf_keys


def query_document(stub, pdf_key: str, user_query: str, top_n: int):
    query_response = stub.QueryDocument(
        intellidocs_v2_pb2.QueryDocumentRequest(
            pdf_key=pdf_key,
            user_query=user_query,
            top_n=top_n
        )
    )
    print(f"Top {top_n} Results for Query: '{user_query}'")
    for result in query_response.results:
        print(f"Chunk: {result.chunk}")
        print(f"Score: {result.score}")
        print("-" * 50)


def run():
    parser = argparse.ArgumentParser(description="IntelliDocs gRPC Client")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    process_parser = subparsers.add_parser("process", help="Process PDFs")
    process_parser.add_argument("pdf_paths", nargs="*", help="PDFs to process")

    query_parser = subparsers.add_parser("query", help="Query a document")
    query_parser.add_argument("pdf_key", help="Unique key of the PDF to query")
    query_parser.add_argument("user_query", help="User query string")
    query_parser.add_argument("--top_n", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = intellidocs_v2_pb2_grpc.IntellidocsServiceStub(channel)

        if args.command == "process":
            pdf_keys = process_pdfs(stub, args.pdf_paths)
            print("You can query documents using the following keys:")
            for pdf_path, pdf_key in pdf_keys.items():
                print(f"{pdf_path}: {pdf_key}")
        elif args.command == "query":
            query_document(stub, args.pdf_key, args.user_query, args.top_n)
        else:
            parser.print_help()


if __name__ == '__main__':
    run()