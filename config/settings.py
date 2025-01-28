import os


class Config:
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
    MYSQL_HOST = os.getenv('DB_HOST')
    MYSQL_USER = os.getenv('DB_USER')
    MYSQL_PASSWORD = os.getenv('DB_PW')
    MYSQL_DB = os.getenv('DB_NAME')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
