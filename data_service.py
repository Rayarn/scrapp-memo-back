import json
import os
from datetime import datetime

DATA_DIR = "/data"



def _ensure():
    os.makedirs(DATA_DIR, exist_ok=True)


def _path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def save_scraped(data: list[dict]):
    _ensure()
    with open(_path("scraped_data.json"), "w", encoding="utf-8") as f:
        json.dump({"data": data, "updated_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)


def load_scraped() -> list[dict]:
    p = _path("scraped_data.json")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("data", [])


def save_analysed(data: list[dict]):
    _ensure()
    with open(_path("analysed_data.json"), "w", encoding="utf-8") as f:
        json.dump({"data": data, "updated_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)


def load_analysed() -> list[dict]:
    p = _path("analysed_data.json")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("data", [])
