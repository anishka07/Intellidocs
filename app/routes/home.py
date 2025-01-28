from flask import Blueprint, render_template
from app.utils.decorators import login_required

home_bp = Blueprint('home', __name__)


@home_bp.route('/')
@login_required
def home():
    return render_template('index.html')


@home_bp.route('/test')
def test_template():
    return render_template('index.html')
