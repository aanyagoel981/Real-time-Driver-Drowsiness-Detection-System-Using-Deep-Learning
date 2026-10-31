"""
Real-time driver drowsiness detection with beep alarm.

  * Eyes  : CNN trained in Colab (drowsiness_model.keras) on landmark-based eye crops
  * Yawns : mouth aspect ratio (MAR) from 68 facial landmarks (OpenCV Facemark LBF)

Put this file in the same folder as:
    drowsiness_model.keras
    beep-warning-6387.mp3
    lbfmodel.yaml
Run:  python drowsiness_webcam.py      (press q to quit)
"""
import os
import time

import cv2
import numpy as np
import pygame
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "drowsiness_model.keras")
BEEP_PATH = os.path.join(HERE, "beep-warning-6387.mp3")
LBF_PATH = os.path.join(HERE, "lbfmodel.yaml")

# CNN output order (confirmed for Closed/Open on the training data)
CLOSED, OPEN = 2, 3
IMG = 145

# Tuning knobs
EYE_CLOSED_SECONDS = 1.5   # eyes closed this long -> alarm
YAWN_SECONDS = 1.2         # mouth wide open this long -> alarm
CLOSED_THRESH = 0.5        # CNN probability that eyes are closed
YAWN_MAR_THRESH = 0.30     # mouth aspect ratio: closed ~0.05, wide yawn ~0.4

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))


def prep(img):
    return cv2.resize(img, (IMG, IMG)).astype("float32") / 255.0


def eye_crop(frame, pts, idx, scale=1.9):
    p = pts[idx]
    cx, cy = p.mean(axis=0)
    size = max((p[:, 0].max() - p[:, 0].min()) * scale, 20)
    x0, y0 = int(max(cx - size / 2, 0)), int(max(cy - size / 2, 0))
    x1, y1 = int(min(cx + size / 2, frame.shape[1])), int(min(cy + size / 2, frame.shape[0]))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None, (x0, y0, x1, y1)
    return frame[y0:y1, x0:x1], (x0, y0, x1, y1)


def mouth_aspect_ratio(pts):
    d = lambda a, b: np.linalg.norm(pts[a] - pts[b])
    return (d(61, 67) + d(62, 66) + d(63, 65)) / (2 * d(60, 64) + 1e-6)


def main():
    if not os.path.exists(LBF_PATH):
        raise SystemExit("lbfmodel.yaml is missing from this folder (see the setup steps).")

    model = tf.keras.models.load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(LBF_PATH)

    pygame.mixer.init()
    pygame.mixer.music.load(BEEP_PATH)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Close other apps using it and try again.")

    closed_since = None
    yawn_since = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120))

        status = "No face detected"
        closed_p = mar = None
        alarm = False

        if len(faces) > 0:
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            found, landmarks = facemark.fit(gray, np.array([face]))
            if found:
                pts = landmarks[0][0]

                # --- eyes: CNN on landmark-based crops ---
                crops = []
                for idx in (LEFT_EYE, RIGHT_EYE):
                    crop, (x0, y0, x1, y1) = eye_crop(frame, pts, idx)
                    if crop is not None:
                        crops.append(prep(crop))
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 200, 0), 1)
                if crops:
                    preds = model(np.array(crops), training=False).numpy()
                    closed_p = float(np.mean(
                        [p[CLOSED] / (p[CLOSED] + p[OPEN] + 1e-6) for p in preds]))

                # --- yawn: mouth aspect ratio ---
                mar = mouth_aspect_ratio(pts)
                for i in range(60, 68):
                    cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 1, (0, 0, 255), -1)

                now = time.time()
                if closed_p is not None and closed_p > CLOSED_THRESH:
                    closed_since = closed_since or now
                else:
                    closed_since = None
                if mar > YAWN_MAR_THRESH:
                    yawn_since = yawn_since or now
                else:
                    yawn_since = None

                if closed_since and now - closed_since >= EYE_CLOSED_SECONDS:
                    alarm = True
                    status = "DROWSY: EYES CLOSED"
                elif yawn_since and now - yawn_since >= YAWN_SECONDS:
                    alarm = True
                    status = "DROWSY: YAWNING"
                else:
                    status = "Alert"
            else:
                status = "Face found, landmarks failed"
                closed_since = yawn_since = None
        else:
            closed_since = yawn_since = None

        # Beep while the alarm condition holds
        if alarm and not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)
        elif not alarm and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

        color = (0, 0, 255) if alarm else (0, 200, 0)
        cv2.putText(frame, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if closed_p is not None and mar is not None:
            cv2.putText(frame, f"eyes closed: {closed_p:.2f}   mouth open (MAR): {mar:.2f}",
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Driver Drowsiness Detection (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pygame.mixer.music.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
