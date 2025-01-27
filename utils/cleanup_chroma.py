import os
import shutil

from utils.constants import PathSettings

if __name__ == '__main__':
    intellidocs_db_dir = os.path.join(PathSettings.CHROMA_DB_PATH, "id_chroma_db", "intellidocs_db")
    if os.path.exists(intellidocs_db_dir):
        shutil.rmtree(intellidocs_db_dir)
        print(f"Removed {intellidocs_db_dir}")

    cache_dir = os.path.join(PathSettings.CHROMA_DB_PATH, "id_chroma_db", "cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Removed {cache_dir}")