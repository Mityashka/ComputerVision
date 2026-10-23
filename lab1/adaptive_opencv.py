"""адаптивная бинаризация (mean) с помощью OpenCV"""
import time

import cv2

from common import BLOCK, C, RESULTS, load_gray


def threshold_opencv(gray, block_size=BLOCK, c=C):
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size должен быть нечётным и >= 3")

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )


def main():
    gray = load_gray()
    ms = float("inf")
    for _ in range(10):
        t = time.perf_counter()
        result = threshold_opencv(gray)
        ms = min(ms, (time.perf_counter() - t) * 1000)
    cv2.imwrite(str(RESULTS / "opencv_result.png"), result)
    print(f"Изображение {gray.shape[1]}x{gray.shape[0]}, окно {BLOCK}, C={C}")
    print(f"OpenCV: {ms:.3f} мс, результат сохранён в results/opencv_result.png")


if __name__ == "__main__":
    main()
