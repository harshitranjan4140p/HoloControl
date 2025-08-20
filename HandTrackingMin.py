import cv2
import mediapipe as mp
import numpy as np
import math
import time
import mss

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


sct = mss.mss()
monitor = sct.monitors[1]


cap = cv2.VideoCapture(0)
show_hud = True  # HUD initially ON
last_toggle_time = 0  # for cooldown

def euclidean_dist(p1, p2, w, h):
    """Calculate pixel distance between two landmarks."""
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.hypot(x2 - x1, y2 - y1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            lm = handLms.landmark
            thumb_tip = lm[4]   # Thumb tip
            index_tip = lm[8]   # Index tip

            # Check pinch
            dist = euclidean_dist(thumb_tip, index_tip, w, h)
            cv2.putText(frame, f"Pinch Dist: {int(dist)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if dist < 40:  # pinch threshold (adjust for your camera)
                now = time.time()
                if now - last_toggle_time > 1:  # 1 sec cooldown
                    show_hud = not show_hud
                    last_toggle_time = now

    # Draw HUD if enabled
    if show_hud:
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (400, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.putText(frame, "HUD Active", (60, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)


        #Windows HUD
        screenshot = np.array(sct.grab(monitor))
        screen = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        h, w, _ = frame.shape

        # Resize screen to almost full size of webcam feed
        PcHUD = cv2.resize(screen, (int(w * 0.9), int(h * 0.9)))  # 90% of webcam
        oh, ow, _ = PcHUD.shape

        # Center it
        x = (w - ow) // 2
        y = (h - oh) // 2

        # Region of interest on webcam
        roi = frame[y:y + oh, x:x + ow]

        # Blend overlay with background
        blended = cv2.addWeighted(PcHUD, 0.7, roi, 1 - 0.7, 0)

        # Place overlay on top
        frame[y:y + oh, x:x + ow] = blended

    cv2.imshow("Hand Gesture HUD", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
