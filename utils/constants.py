import os


class PathSettings:
    PROJECT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PDF_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'pdfs')
    CHROMA_DB_PATH = os.path.join(PROJECT_DIR_PATH, 'id_chroma_db')
    CACHE_DIR_PATH = os.path.join(CHROMA_DB_PATH, 'cache')


class ConstantSettings:
    CHUNK_SIZE: int = 400
    EMBEDDING_MODEL_NAME: str = 'all-mpnet-base-v2'
    GEMINI_PROMPT = """ From the given context answer the user query. Give answer in a concise form.
    context: {}
    user query: {}
    If the context does not match the query, notify the user with the message.
    """
    SPACY_LOAD: str = "en_core_web_sm"
