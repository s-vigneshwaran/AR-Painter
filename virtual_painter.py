import cv2
import numpy as np
import handtrackingmodule as htm
import os

assets = []
for asset in os.listdir('assets'):
    assets.append(cv2.imread('assets/' + str(asset)))
tool_picker = assets[0]

draw_color = (255, 0, 255)
brush_thickness = 15
eraser_thickness = 30

cap = cv2.VideoCapture(0)
w_cam, h_cam = 1280, 720
cap.set(3, w_cam)
cap.set(4, h_cam)

detector = htm.HandDetector(detection_con=0.85, max_hands=1)
xp, yp = 0, 0
# canvas = np.full((720, 1280, 3), 255, np.uint8)
canvas = np.zeros((h_cam, w_cam, 3), np.uint8)

while True:
    # Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find Landmarks
    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index Finger
        x2, y2 = lmList[12][1:]  # Middle Finger

        # Check active Fingers
        fingers = detector.active_fingers()

        # Selection Mode - two fingers
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            # Checking Controls
            if y1 < 125:
                if 250 < x1 < 450:
                    tool_picker = assets[0]
                    draw_color = (255, 0, 255)
                elif 550 < x2 < 750:
                    tool_picker = assets[1]
                    draw_color = (255, 0, 0)
                elif 800 < x2 < 950:
                    tool_picker = assets[2]
                    draw_color = (0, 255, 0)
                elif 1050 < x2 < 1200:
                    tool_picker = assets[3]
                    draw_color = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)

        # Drawing Mode - Index Fingers
        if fingers[1] and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if draw_color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)

            xp, yp = x1, y1

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inverse = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inverse = cv2.cvtColor(img_inverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inverse)
    img = cv2.bitwise_or(img, canvas)

    img[0:125, 0:1280] = tool_picker

    # img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
    cv2.imshow('Virtual Painter', img)
    # cv2.imshow('Canvas', canvas)
    key = cv2.waitKey(1)
    if key == 27:
        break
