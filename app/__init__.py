from dotenv import load_dotenv
from flask import Flask, render_template
from flask_mysqldb import MySQL
from flask_socketio import SocketIO
import os

from config.settings import Config

load_dotenv()

mysql = MySQL()
socket_message = SocketIO()


def create_app():
    # Get the absolute path to the app directory
    app_dir = os.path.dirname(os.path.abspath(__file__))

    app = Flask(__name__,
                template_folder='templates',  # Changed to relative path
                static_folder='static')

    app.config.from_object(Config)

    # Initialize extensions
    mysql.init_app(app)
    socket_message.init_app(app)

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.home import home_bp
    from app.routes.chat import chat_bp
    from app.routes.upload import upload_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(home_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(upload_bp)

    return app
