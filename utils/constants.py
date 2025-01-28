import os


class PathSettings:
    PROJECT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PDF_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'pdfs')
    CHROMA_DB_PATH = os.path.join(PROJECT_DIR_PATH, 'id_chroma_db')
    PICKLE_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'pickle_models')
    CSV_DB_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'CSV_db')
    UPLOADS_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'uploads')


class ConstantSettings:
    CHUNK_SIZE: int = 100
    CHROMA_DB_COLLECTION = 'intellidocs_db'
    EMBEDDING_MODEL_NAME: str = 'all-mpnet-base-v2'
    LLM_MODEL_NAME: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    GEMINI_PROMPT = """ You are a RAG system named Intellidocs. The user using you has this query: {}\n. And this is the query the RAG has generated
    {}\n. Your job is to generate a response on the query based on the context. Make the context similar to the RAG response. Greet me as Intellidocs and answer the question.
    """
    GEMINI_NC_PROMPT: str = 'Just respond to the query given to you by the user. This is the query the user sent you: {}'
    ALLOWED_EXTENSIONS = {'pdf'}
    SPACY_LOAD: str = "en_core_web_sm"


if __name__ == '__main__':
    print(PathSettings.CSV_DB_DIR_PATH)
