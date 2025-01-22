import os


class PathSettings:
    PROJECT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PDF_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'pdfs')
    CHROMA_DB_PATH = os.path.join(PROJECT_DIR_PATH, 'db')
    PICKLE_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'pickle_models')
    CSV_DB_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'CSV_db')
    UPLOADS_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'uploads')


class ConstantSettings:
    EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
    LLM_MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    GEMINI_PROMPT = """ You are a RAG system named Intellidocs. The user using you has this query: {}\n. And this is the query the RAG has generated
    {}\n. Your job is to generate a response on the query based on the context. And if the context does not make sense, you are supposed to make 
    your own structured and true response.\n What ever happens, do not mention anything about the users context not being related to the questions.
    Also dont make the response too long. make it short and 2 paragraphs at max.
    """
    GEMINI_NC_PROMPT = 'Just respond to the query given to you by the user. This is the message the user sent you: {}'
    ALLOWED_EXTENSIONS = {'pdf'}


if __name__ == '__main__':
    print(PathSettings.CSV_DB_DIR_PATH)
