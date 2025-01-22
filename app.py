import os
from functools import wraps

from flask import Flask, render_template, request, session, flash, redirect, url_for
from flask_socketio import SocketIO, emit
from flask_mysqldb import MySQL
import MySQLdb.cursors
import bcrypt
from dotenv import load_dotenv

load_dotenv()

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

mysql = MySQL(app)
socket_message = SocketIO(app)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('You need to log in first!', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


@socket_message.on('send_message')
def handle_send_message_event(data):
    message = data['message']
    username = session['username'] if 'username' in session else 'Guest'
    emit('receive_message', {'message': message, 'username': username}, broadcast=True)


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