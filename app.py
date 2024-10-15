import os
from functools import wraps
from flask import Flask, render_template, request, session, flash, redirect, url_for
from flask_mysqldb import MySQL
import MySQLdb.cursors
import bcrypt
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')

app.config['MYSQL_HOST'] = os.getenv('DB_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('DB_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('DB_PW', 'anishka123')
app.config['MYSQL_DB'] = os.getenv('DB_NAME', 'intellidocs_login')

mysql = MySQL(app)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('You need to log in first!', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


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


@app.route('/logout')
@login_required
def logout():
    session.pop('username', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
