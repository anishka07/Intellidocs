import os


class PathSettings:
    PROJECT_DIR_PATH = os.path.dirname(os.getcwd())
    PDF_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'pdfs')
    CHROMA_DB_PATH = os.path.join(PROJECT_DIR_PATH, 'db')
    LLM_PATH = os.path.join(PROJECT_DIR_PATH, 'LaMini-T5-738M')
    PICKLE_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'pickle_models')
    CSV_DB_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'CSV_db')


class ConstantSettings:
    EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
