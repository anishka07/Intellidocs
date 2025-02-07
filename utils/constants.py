import os


class PathSettings:
    PROJECT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PDF_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'pdfs')
    CHROMA_DB_PATH = os.path.join(PROJECT_DIR_PATH, 'id_chroma_db')
    UPLOADS_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'uploads')
    CACHE_DIR_PATH = os.path.join(CHROMA_DB_PATH, 'cache')


class ConstantSettings:
    CHUNK_SIZE: int = 100
    CHROMA_DB_COLLECTION = 'intellidocs_db'
    EMBEDDING_MODEL_NAME: str = 'all-mpnet-base-v2'
    LLM_MODEL_NAME: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    GEMINI_PROMPT = """ You are a RAG. This is the context provided to you by the user: {}\n. This is the query provided to you by the user: {}\n.
    Now create a response based on that context. Structure the context nicely and don't give any extra answers.
    """
    ALLOWED_EXTENSIONS = {'pdf'}
    SPACY_LOAD: str = "en_core_web_sm"
