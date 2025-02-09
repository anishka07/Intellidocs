import os


class PathSettings:
    PROJECT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PDF_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'pdfs')
    CHROMA_DB_PATH = os.path.join(PROJECT_DIR_PATH, 'id_chroma_db')
    UPLOADS_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'uploads')
    CACHE_DIR_PATH = os.path.join(CHROMA_DB_PATH, 'cache')


class ConstantSettings:
    CHUNK_SIZE: int = 400
    EMBEDDING_MODEL_NAME: str = 'all-mpnet-base-v2'
    LLM_MODEL_NAME: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    GEMINI_PROMPT = """ You are a RAG. This is the context provided to you by the user: {}\n. This is the query provided to you by the user: {}\n.
    Now summarize the content given to you in 2 paragraphs and only use the points in the context given to you that makes sense. make bullet points as well 
    Don't mention the provided text. give your independent response. give a conclusion as well that summarizes the content in easy language.
    """
    ALLOWED_EXTENSIONS = {'pdf'}
    SPACY_LOAD: str = "en_core_web_sm"
