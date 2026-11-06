#%%
import os
import time
import cv2
import numpy as np

#%%
VIDEOS = ["test-videos/mona-lisa.avi", "test-videos/mona-lisa-blur.avi", "test-videos/mona-lisa-blur-extra-credit.avi", "test-videos/20261106_181427.mp4"]
LABEL = "Object"
RATIO = 0.75
MIN_INLIERS = 12
RANSAC_THRESH = 5.0
#%%
def track(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"ошибка пути {video_path}")
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Не удалось прочитать первый кадр")
    h, w = first.shape[:2]

    #Первый кадр
    sift = cv2.SIFT_create()
    kp_t, des_t = sift.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), None)
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    print(f"{video_path}: ключевых точек: {len(kp_t)}")

    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"не удалось открыть на видео {out_path}")

    total = found = 0
    t_start = time.time()
    frame = first
    while ok:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        quad = None
        if des is not None and len(kp) >= 2:
            pairs = matcher.knnMatch(des_t, des, k=2)
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
            if len(good) >= 4:
                src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
                if H is not None and mask.sum() >= MIN_INLIERS:
                    q = cv2.perspectiveTransform(corners, H)
                    if cv2.isContourConvex(q.astype(np.float32)):
                        quad = q

        if quad is not None:
            pts = np.int32(quad).reshape(-1, 2)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)
            x, y = pts[np.argmin(pts[:, 1])]
            cv2.putText(frame, LABEL, (int(x), max(int(y) - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            found += 1

        out.write(frame)
        total += 1
        ok, frame = cap.read()

    cap.release()
    out.release()
    dt = time.time() - t_start
    print(f"  кадров: {total}, объект найден: {found} ({100 * found / total:.1f}%), "
          f"{total / dt:.1f} FPS: {out_path}")
#%%
for v in VIDEOS:
    track(v, "out_" + os.path.basename(v).rsplit(".", 1)[0] + ".mp4")