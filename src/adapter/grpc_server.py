import grpc 
import logging
from concurrent import futures

from src.intellidocspb import intellidocs_pb2
from src.intellidocspb import intellidocs_pb2_grpc

from src.model.intellidocs_rag.intellidocs_main import IntellidocsRAG  


class IntelliDocsServiceServicer(intellidocs_pb2_grpc.IntelliDocsServiceServicer):

    def __init__(self):
        self.rag = IntellidocsRAG()

    def ProcessDocuments(self, request, context):
        try:
            self.rag.process_documents(request.document_paths)
            return intellidocs_pb2.ProcessDocumentResponse(
                message="Documents processed successfully!"
            )
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return intellidocs_pb2.ProcessDocumentResponse(
                message=f"Failed: {str(e)}"
            )
        
    def RetrieveTopN(self, request, context):
        try:
            results = self.rag.retrieve_top_n(
                user_query=request.user_query,
                doc_key=request.doc_key,
                top_n=request.top_n
            )
            chunks = [
                intellidocs_pb2.RetrievedChunks(chunks=r["chunk"], score=r["score"])
                for r in results
            ]
            return intellidocs_pb2.RetrieveTopNResponse(chunks=chunks)
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return intellidocs_pb2.RetrieveTopNResponse()
        

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    intellidocs_pb2_grpc.add_IntelliDocsServiceServicer_to_server(
        IntelliDocsServiceServicer(), server=server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Intellidocs server runnning...")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
