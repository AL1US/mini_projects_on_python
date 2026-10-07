from pathlib import Path
import json

def get_data(file_path: Path):
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    return data
