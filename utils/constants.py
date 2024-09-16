import os


class PathSettings:
    PROJECT_DIR_PATH = os.path.dirname(os.getcwd())
    PDF_FILE_PATH = os.path.join(PROJECT_DIR_PATH, 'pdfs')


class ConstantSettings:
    EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
