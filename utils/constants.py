import os


class PathSettings:
    PROJECT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PDF_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'pdfs')
    CHROMA_DB_PATH = os.path.join(PROJECT_DIR_PATH, 'db')
    PICKLE_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'pickle_models')
    CSV_DB_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'CSV_db')


class ConstantSettings:
    EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
    LLM_MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
