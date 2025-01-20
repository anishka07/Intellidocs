import os
import shutil
from constants import PathSettings


def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


if __name__ == '__main__':
    try:
        clear_directory(PathSettings.CSV_DB_DIR_PATH)
        print("Successfully cleared CSV database directory.")
    except Exception as e:
        print("Exception occurred while cleaning the directory: ", e)
