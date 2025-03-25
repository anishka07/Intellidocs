import grpc
from concurrent import futures

from src.intellidocspb.v2 import intellidocs_v2_pb2
from src.intellidocspb.v2 import intellidocs_v2_pb2_grpc
from src.model.intellidocs_rag.intellidocs_main import IntellidocsRAG
from utils.constants import ConstantSettings, PathSettings


class IntellidocsService(intellidocs_v2_pb2_grpc.IntellidocsServiceServicer):

    def __init__(self):
        self.rag = IntellidocsRAG(
            chunk_size=ConstantSettings.CHUNK_SIZE,
            embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
            chroma_db_dir=PathSettings.CHROMA_DB_PATH
        )

    def ProcessPDFs(self, request, context):
        pdf_paths = request.pdf_paths
        self.rag.process(pdf_paths)
        pdf_keys = {pdf_path: self.rag._generate_document_key(pdf_path) for pdf_path in pdf_paths}
        return intellidocs_v2_pb2.ProcessPDFsResponse(pdf_keys=pdf_keys)

    def QueryDocument(self, request, context):
        pdf_key = request.pdf_key
        user_query = request.user_query
        top_n = request.top_n
        results = self.rag.retrieve_top_n(
            doc_key=pdf_key,
            user_query=user_query,
            top_n=top_n,
        )
        response = intellidocs_v2_pb2.QueryDocumentResponse()
        for result in results:
            response.results.append(
                intellidocs_v2_pb2.QueryDocumentResponse.QueryResult(
                    chunk=result['chunk'],
                    score=result['score'],
                )
            )
        return response


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    intellidocs_v2_pb2_grpc.add_IntellidocsServiceServicer_to_server(IntellidocsService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
