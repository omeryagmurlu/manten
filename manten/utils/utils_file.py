import json
import os
from pathlib import Path


def touch(path):
    with Path(path).open("a"):
        os.utime(path, None)


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write(path, content):
    with Path(path).open("w") as f:
        f.write(content)


def write_json(path, content):
    with Path(path).open("w") as f:
        json.dump(content, f)
