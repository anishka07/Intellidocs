from utils.constants import ConstantSettings


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ConstantSettings.ALLOWED_EXTENSIONS
