import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


# MEDIAPIPE HAND SETUP

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# CAMERA
cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0

draw_color = (0, 0, 255)   # RED pen
erase_color = (0, 0, 0)         # erase background

# TKINTER WINDOW
root = tk.Tk()
root.title("Gesture Notebook AI")

frame = tk.Frame(root)
frame.pack()

notebook_label = tk.Label(frame)
notebook_label.pack(side="left")

webcam_label = tk.Label(frame)
webcam_label.pack(side="right")

# FINGER DETECTION
def fingers_up(hand):

    tips = [8, 12, 16, 20]
    fingers = []

    for tip in tips:
        if hand.landmark[tip].y < hand.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# MAIN LOOP
def update():

    global canvas, prev_x, prev_y

    ret, frame_cam = cap.read()
    frame_cam = cv2.flip(frame_cam, 1)

    h, w, _ = frame_cam.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame_cam,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            fingers = fingers_up(hand)

            index_up = fingers[0] == 1
            fist = sum(fingers) == 0

            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            #  DRAW MODE
            if index_up and not fist:

                # pen cursor
                cv2.circle(frame_cam, (x, y), 8, (255, 255, 255), -1)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (x, y),
                    draw_color,
                    5
                )

                prev_x, prev_y = x, y

            # ERASER MODE 
            elif fist:

                # erase canvas
                cv2.circle(
                    canvas,
                    (x, y),
                    35,
                    erase_color,
                    -1
                )

                # red eraser cursor
                cv2.circle(
                    frame_cam,
                    (x, y),
                    35,
                    (0, 0, 255),
                    3
                )

                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0

    # DISPLAY 
    notebook_view = canvas.copy()

    notebook_rgb = cv2.cvtColor(
        notebook_view,
        cv2.COLOR_BGR2RGB
    )

    webcam_rgb = cv2.cvtColor(
        frame_cam,
        cv2.COLOR_BGR2RGB
    )

    notebook_img = ImageTk.PhotoImage(
        Image.fromarray(notebook_rgb).resize((700, 500))
    )

    webcam_img = ImageTk.PhotoImage(
        Image.fromarray(webcam_rgb).resize((250, 180))
    )

    notebook_label.config(image=notebook_img)
    notebook_label.image = notebook_img

    webcam_label.config(image=webcam_img)
    webcam_label.image = webcam_img

    root.after(10, update)


# =============================
# START APPLICATION
# =============================
update()
root.mainloop()
