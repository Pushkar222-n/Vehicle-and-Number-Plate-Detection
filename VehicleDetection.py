import cv2
import numpy as np


cap = cv2.VideoCapture(r"C:\Users\KIIT01\Documents\ML Projects\OpenCV Practice\Speed Tracking\test.mp4")
_, prev_frame = cap.read()
prev = cv2.resize(prev_frame, (0, 0), fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
subtractor = cv2.createBackgroundSubtractorMOG2()


while True:
    kernel = np.ones((4, 4), np.uint8)
    ret, frame_1 = cap.read()

    if ret:
        r, c, ch = frame_1.shape
        frame = cv2.resize(frame_1, (0, 0), fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)

        mask = subtractor.apply(frame)
        # morphological transformation
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel)
        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(
            mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


        for contour in contours:
            if cv2.contourArea(contour) < 340:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


        cv2.imshow("original frame", frame)
        cv2.imshow("mask", mask)
        key = cv2.waitKey(30)
        if key == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()