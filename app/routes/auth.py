from flask import Blueprint, render_template, request, session, redirect, url_for, flash
from app import mysql
import bcrypt

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        account_pw = request.form['password'].encode('utf-8')
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account and bcrypt.checkpw(account_pw, account[1].encode('utf-8')):
            session['username'] = account[0]
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home.home'))
        flash('Incorrect username or password!', 'danger')
    return render_template('auth/login.html')


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        confirm_password = request.form['confirm_password'].encode('utf-8')

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
        else:
            cursor = mysql.connection.cursor()
            cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
            account = cursor.fetchone()
            if account:
                flash('Account already exists!', 'danger')
            else:
                hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())
                cursor.execute('INSERT INTO accounts (username, password) VALUES (%s, %s)',
                               (username, hashed_password.decode('utf-8')))
                mysql.connection.commit()
                flash('Successfully registered!', 'success')
                return redirect(url_for('auth.login'))
    return render_template('auth/signup.html')


@auth_bp.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully!', 'info')
    return redirect(url_for('auth.login'))
