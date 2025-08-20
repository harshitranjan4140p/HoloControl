import cv2
import mediapipe as mp
import numpy as np
import math
import time
import mss

# -------------------- Mediapipe Setup --------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# -------------------- Screen Capture Setup --------------------
sct = mss.mss()
monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]

# -------------------- State --------------------
show_hud = True
last_toggle_time = 0

# -------------------- Helper --------------------
def euclidean_dist(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.hypot(x2 - x1, y2 - y1)

# -------------------- Main Loop --------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Reduce webcam resolution for performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_screen_time = 0
screen_bgr = None
screen_interval = 1 / 10.0  # capture desktop only 10 FPS

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Hand + pinch detection
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            lm = handLms.landmark
            thumb_tip, index_tip = lm[4], lm[8]
            dist = euclidean_dist(thumb_tip, index_tip, w, h)

            cv2.putText(frame, f"Pinch: {int(dist)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if dist < 40:
                now = time.time()
                if now - last_toggle_time > 1.0:
                    show_hud = not show_hud
                    last_toggle_time = now

    # HUD
    if show_hud:
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (w - 50, h - 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Limit screen grabs to ~10 FPS
        now = time.time()
        if screen_bgr is None or (now - last_screen_time) >= screen_interval:
            screenshot = np.array(sct.grab(monitor))
            screen_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            last_screen_time = now

        if screen_bgr is not None:
            PcHUD = cv2.resize(screen_bgr, (int(w * 0.9), int(h * 0.9)))
            oh, ow, _ = PcHUD.shape
            x, y = (w - ow) // 2, (h - oh) // 2

            if y + oh <= h and x + ow <= w:
                roi = frame[y:y + oh, x:x + ow]
                blended = cv2.addWeighted(PcHUD, 0.7, roi, 0.3, 0)
                frame[y:y + oh, x:x + ow] = blended

        cv2.putText(frame, "HUD Active (pinch to toggle)",
                    (60, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    cv2.imshow("HoloControl Wired", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
