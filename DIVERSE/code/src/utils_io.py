import os

def get_file(path):
    if os.path.isdir(path):
        return os.path.join(path, os.listdir(path)[0])
    else:
        return path