from flask import Blueprint, session, render_template
from flask_socketio import emit
from app import socket_message
from app.utils.decorators import login_required
from model.intellidocs_rag_final.id_chroma_rag import IntellidocsRAG
from model.llms.gemini_response import gemini_response
from utils.constants import ConstantSettings, PathSettings

chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/chat')
@login_required
def chat():
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
        rag = IntellidocsRAG(
            pdf_doc_path=pdf_data.get('pdf_path'),
            chunk_size=ConstantSettings.CHUNK_SIZE,
            embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
            chroma_db_dir=PathSettings.CHROMA_DB_PATH
        )
        top_results = rag.retrieve_top_n(message, collection_name, top_n=3)
        context = " ".join([result['chunk'] for result in top_results])
        llm_response = gemini_response(message, context=context)
        emit('receive_message', {'message': message, 'username': username}, broadcast=True)
        emit('receive_message', {'message': llm_response, 'username': 'IntelliDocs'}, broadcast=True)
    except Exception as e:
        emit('receive_message', {'message': "Error processing your request", 'username': 'IntelliDocs'}, broadcast=True)
