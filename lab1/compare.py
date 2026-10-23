
import time

import cv2
import numpy as np

from adaptive_native import threshold_native
from adaptive_opencv import threshold_opencv
from common import BLOCK, C, RESULTS, load_gray


def process():
    gray = load_gray()

    results = {}

    start = time.perf_counter()
    results["native"] = threshold_native(gray)
    print(f"[TIME] Обычный: {time.perf_counter() - start:.4f} сек.")

    start = time.perf_counter()
    results["opencv"] = threshold_opencv(gray)
    print(f"[TIME] OpenCV: {time.perf_counter() - start:.4f} сек.")

    diff = cv2.absdiff(results["native"], results["opencv"])
    print(f"Различающихся пикселей: {np.count_nonzero(diff)}")
    print(f"Параметры: окно {BLOCK}x{BLOCK}, C = {C}")

    cv2.imwrite(str(RESULTS / "native_result.png"), results["native"])
    cv2.imwrite(str(RESULTS / "opencv_result.png"), results["opencv"])
    cv2.imwrite(str(RESULTS / "diff.png"), diff)



if __name__ == "__main__":
    process()
