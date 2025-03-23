import os

import grpc

from intellidocspb.v2 import intellidocs_v2_pb2
from intellidocspb.v2 import intellidocs_v2_pb2_grpc
from utils.constants import PathSettings


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = intellidocs_v2_pb2_grpc.IntellidocsServiceStub(channel)

        pdf_paths = [
            os.path.join(PathSettings.PDF_DIR_PATH, 'monopoly.pdf'),
            os.path.join(PathSettings.PDF_DIR_PATH, 'POM_Unit-1.pdf'),
        ]
        process_response = stub.ProcessPDFs(
            intellidocs_v2_pb2.ProcessPDFsRequest(pdf_paths=pdf_paths)
        )

        print("PDF Keys: ", process_response.pdf_keys)

        pdf_key = 'monopoly_key'
        user_query = "What is the monopoly speed die rules?"
        query_response = stub.QueryDocument(
            intellidocs_v2_pb2.QueryDocumentRequest(
                pdf_key=pdf_key,
                user_query=user_query,
                top_n=5
            )
        )
        print("Query Results: ")
        for result in query_response.results:
            print(f"Chunk: {result.chunk}, Score: {result.score}")


if __name__ == '__main__':
    run()
