from globals import *
import shutil

def run():
    shutil.rmtree(TEMP_DIR)
    shutil.rmtree(TEMP_FILES_DIR)

if __name__ == "__main__":
    run()