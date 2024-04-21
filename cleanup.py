from globals import *
import shutil
import os

def run():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print("Removed ", TEMP_DIR)
    else:
        print(TEMP_DIR, "does not exist")
    if os.path.exists(TEMP_FILES_DIR):
        shutil.rmtree(TEMP_FILES_DIR)
        print("Removed ", TEMP_FILES_DIR)
    else:
        print(TEMP_FILES_DIR, "does not exist")

if __name__ == "__main__":
    run()