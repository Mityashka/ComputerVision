from pathlib import Path

import cv2

BASE = Path(__file__).parent
IMAGES = BASE / "images"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

BLOCK = 11   #размер окна
C = 10       #константа


def find_image() -> Path:
    for name in ("input.jpg", "input.jpeg", "input.png"):
        p = IMAGES / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "файл не в корне"
    )


def load_gray():
    path = find_image()
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"ошибка при чтении {path}")
    return gray
