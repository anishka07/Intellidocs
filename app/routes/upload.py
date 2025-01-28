from flask import Blueprint, request, session, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename
import os
from app.utils.helpers import allowed_file
from app.utils.decorators import login_required
from model.intellidocs_rag_final.id_chroma_rag import IntellidocsRAG
from utils.constants import PathSettings, ConstantSettings

upload_bp = Blueprint('upload', __name__)


@upload_bp.route('/upload_pdf', methods=['GET', 'POST'])
@login_required
def upload_pdf():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            flash('No file part!', 'danger')
            return redirect(request.url)

        file = request.files['pdf_file']
        if file.filename == '':
            flash('No selected file!', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(PathSettings.UPLOADS_DIR_PATH, filename)
            file.save(file_path)

            # Process the PDF using RAG logic
            rag = IntellidocsRAG(
                pdf_doc_path=file_path,
                chunk_size=ConstantSettings.CHUNK_SIZE,
                embedding_model=ConstantSettings.EMBEDDING_MODEL_NAME,
                chroma_db_dir=PathSettings.CHROMA_DB_PATH
            )
            # Additional processing code...

            session['pdf_data'] = {
                'pdf_path': file_path,
                'filename': filename,
                'collection_name': filename
            }
            flash('PDF processed successfully!', 'success')
            return redirect(url_for('chat.chat'))

    return render_template('intellidocschat.html')
