import json
import os


def touch(path):
    with open(path, "a"):
        os.utime(path, None)


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def write(path, content):
    with open(path, "w") as f:
        f.write(content)


def write_json(path, content):
    with open(path, "w") as f:
        json.dump(content, f)
