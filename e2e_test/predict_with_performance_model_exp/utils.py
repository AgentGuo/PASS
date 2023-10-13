import json
import requests


def load_json_file(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        # 将字典保存为JSON
        json.dump(data, f, ensure_ascii=False, indent=4)
