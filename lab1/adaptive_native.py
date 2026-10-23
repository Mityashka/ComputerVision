"""Вариант 2: адаптивная бинаризация (mean), написанная вручную на чистом Python"""
import time
from math import ceil

import cv2
import numpy as np

from adaptive_opencv import threshold_opencv
from common import BLOCK, C, RESULTS, load_gray


def threshold_native(gray, block_size=BLOCK, c=C):
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size должен быть нечётным и >= 3")
    h, w = gray.shape
    r = block_size // 2
    area = block_size * block_size

    padded = np.pad(gray, r, mode="edge").tolist()
    src = gray.tolist()
    out = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            s = 0
            for dy in range(block_size):
                row = padded[y + dy]
                for dx in range(block_size):
                    s += row[x + dx]
            if src[y][x] - round(s / area) > -ceil(c):
                out[y][x] = 255
    return np.array(out, dtype=np.uint8)


def main():
    gray = load_gray()
    print(f"Изображение {gray.shape[1]}x{gray.shape[0]}, окно {BLOCK}, C={C}")

    t = time.perf_counter()
    res = threshold_native(gray)
    ms = (time.perf_counter() - t) * 1000
    cv2.imwrite(str(RESULTS / "native_result.png"), res)
    print(f"вручную на питоне: {ms:.0f} мс, save в results/native_result.png")

    diff = (res != threshold_opencv(gray)).astype(np.uint8) * 255
    cv2.imwrite(str(RESULTS / "diff.png"), diff)
    print(f"Отличие от OpenCV: {int((diff > 0).sum())} пикселей")


if __name__ == "__main__":
    main()
