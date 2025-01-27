import logging
import os
from functools import wraps

import MySQLdb.cursors
import bcrypt
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, flash, redirect, url_for
from flask_mysqldb import MySQL
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

from model.intellidocs_rag_final.id_chroma_rag import IntellidocsRAG
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
                emit('processing_update', {'message': 'Processing PDF...'}, broadcast=True, namespace='/')

                # Initialize IntellidocsRAG with the uploaded PDF
                rag = IntellidocsRAG(
                    pdf_doc_path=file_path,
                    chunk_size=ConstantSettings.CHUNK_SIZE,
                    embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
                    chroma_db_dir=PathSettings.CHROMA_DB_PATH
                )

                # Check if embeddings for this PDF already exist
                collection_name = filename  # Use filename as the collection name
                existing_collection = rag.chroma_client.get_collection(collection_name)
                if existing_collection is not None:
                    flash('Embeddings already exist for this PDF. Using existing embeddings.', 'info')
                else:
                    # Process the PDF and store embeddings
                    extracted_text = rag.extract_text_from_document_fitz()
                    text_chunks = rag.text_chunking(extracted_text)
                    embeddings = rag.generate_embeddings(text_chunks)
                    rag.store_embeddings(text_chunks, embeddings, collection_name)
                    flash('PDF processed and embeddings stored successfully!', 'success')

                # Store session data for retrieval
                session['pdf_data'] = {
                    'pdf_path': file_path,
                    'filename': filename,
                    'collection_name': collection_name
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
    collection_name = pdf_data.get('collection_name')

    if not collection_name:
        emit('receive_message', {'message': "Please upload a PDF first", 'username': 'IntelliDocs'}, broadcast=True)
        return

    try:
        # Initialize RAG with collection name for retrieval
        rag = IntellidocsRAG(
            pdf_doc_path=pdf_data.get('pdf_path'),
            chunk_size=ConstantSettings.CHUNK_SIZE,
            embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
            chroma_db_dir=PathSettings.CHROMA_DB_PATH
        )

        # Retrieve top results using the RAG class
        top_results = rag.retrieve_top_n(user_query=message, chroma_collection_name=collection_name, top_n=3)

        # Combine retrieved chunks for LLM response
        context = " ".join([result['chunk'] for result in top_results])
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
