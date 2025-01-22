import logging
import os
from functools import wraps

import MySQLdb.cursors
import bcrypt
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, flash, redirect, url_for
from flask_mysqldb import MySQL
from flask_socketio import SocketIO, emit
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename

from model.intellidocs_rag_final.chunk_processor import ChunkProcessor
from model.intellidocs_rag_final.pdf_loader import PdfLoader
from model.intellidocs_rag_final.retrieval_process import Retriever
from model.llms.gemini_response import gemini_response
from utils.constants import PathSettings, ConstantSettings

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')

mysql_host = os.getenv('DB_HOST')
mysql_user = os.getenv('DB_USER')
mysql_password = os.getenv('DB_PW')
mysql_db_name = os.getenv('DB_NAME')

app.config['MYSQL_HOST'] = mysql_host
app.config['MYSQL_USER'] = mysql_user
app.config['MYSQL_PASSWORD'] = mysql_password
app.config['MYSQL_DB'] = mysql_db_name
app.config['UPLOAD_FOLDER'] = PathSettings.UPLOADS_DIR_PATH

mysql = MySQL(app)
socket_message = SocketIO(app)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ConstantSettings.ALLOWED_EXTENSIONS


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('You need to log in first!', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


@app.route('/upload_pdf', methods=['GET', 'POST'])
@login_required
def upload_pdf():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['pdf_file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                logger.info(f"Starting PDF processing for {filename}")
                emit('processing_update', {'message': 'Loading PDF...'}, broadcast=True, namespace='/')

                pdf_loader = PdfLoader(pdf_path=file_path)
                pages_and_texts = pdf_loader.add_tokenized_sentences()
                logger.info(f"PDF loaded and tokenized: {len(pages_and_texts)} pages")
                emit('processing_update', {'message': f'PDF loaded: {len(pages_and_texts)} pages'}, broadcast=True,
                     namespace='/')

                logger.info("Processing chunks...")
                emit('processing_update', {'message': 'Processing text chunks...'}, broadcast=True, namespace='/')
                chunk_processor = ChunkProcessor(pages_and_texts=pages_and_texts, min_token_length=20)
                filtered_chunks = chunk_processor.filter_chunks_by_token_length()
                logger.info(f"Chunks processed: {len(filtered_chunks)} chunks")

                logger.info("Generating embeddings...")
                emit('processing_update', {'message': 'Generating embeddings...'}, broadcast=True, namespace='/')
                chunk_df = pd.DataFrame(filtered_chunks)
                embeddings = chunk_processor.data_frame['sentence_chunk'].apply(
                    lambda x: SentenceTransformer('all-mpnet-base-v2').encode(x).tolist()
                )
                chunk_df['embeddings'] = embeddings
                chunk_df['model_name'] = 'all-mpnet-base-v2'

                embeddings_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_embeddings.csv")
                chunk_df.to_csv(embeddings_path, index=False)
                logger.info("Embeddings saved")

                session['pdf_data'] = {
                    'embeddings_path': embeddings_path,
                    'pdf_path': file_path,
                    'filename': filename,
                    'num_pages': len(pages_and_texts),
                    'num_chunks': len(filtered_chunks)
                }

                emit('processing_update', {'message': 'PDF processing complete!'}, broadcast=True, namespace='/')
                return redirect(url_for('chat'))
            except Exception as e:
                logger.error(f"Error processing PDF: {e}")
                emit('processing_update', {'message': f'Error: {str(e)}'}, broadcast=True, namespace='/')
                flash('Error processing PDF file', 'danger')
                return redirect(request.url)

    return render_template('intellidocschat.html')


@socket_message.on('send_message')
def handle_send_message_event(data):
    message = data['message']
    username = session.get('username', 'Guest')
    pdf_data = session.get('pdf_data', {})
    embeddings_path = pdf_data.get('embeddings_path')

    if not embeddings_path:
        emit('receive_message', {'message': "Please upload a PDF first", 'username': 'IntelliDocs'}, broadcast=True)
        return

    try:
        retriever = Retriever(embeddings_df_path=embeddings_path)
        scores, indices = retriever.retrieve_relevant_resources(query=message, n_resources_to_return=3)

        relevant_chunks = [retriever.pages_and_chunks[idx]["sentence_chunk"] for idx in indices]
        context = " ".join(relevant_chunks)

        llm_response = gemini_response(message, context=context)

        emit('receive_message', {'message': message, 'username': username}, broadcast=True)
        emit('receive_message', {'message': llm_response, 'username': 'IntelliDocs'}, broadcast=True)
    except Exception as e:
        logger.error(f"Error in message processing: {e}")
        emit('receive_message', {'message': "Error processing your request", 'username': 'IntelliDocs'}, broadcast=True)

@app.route('/')
@login_required
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        account_pw = request.form['password'].encode('utf-8')

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()

        if account and bcrypt.checkpw(account_pw, account['password'].encode('utf-8')):
            session['username'] = account['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        flash('Incorrect username or password!', 'danger')

    return render_template('auth/login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        confirm_password = request.form['confirm_password'].encode('utf-8')

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
        else:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
            account = cursor.fetchone()

            if account:
                flash('Account already exists!', 'danger')
            else:
                hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())
                cursor.execute('INSERT INTO accounts (username, password) VALUES (%s, %s)',
                               (username, hashed_password.decode('utf-8')))
                mysql.connection.commit()
                flash('You have successfully registered!', 'success')
                return redirect(url_for('login'))

    return render_template('auth/signup.html')


@app.route('/chat')
@login_required
def chat():
    return render_template('intellidocschat.html')


@app.route('/logout')
@login_required
def logout():
    session.pop('username', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
