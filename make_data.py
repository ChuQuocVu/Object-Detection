from cv2 import cv2
import os.path
from os import path
import time

object_label = "50000"

i = 0

cap = cv2.VideoCapture(0)

while True:
    i += 1
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if (i >= 60) and (i <= 1060):
        print("Frame captrued: ", i-60)
        if not os.path.exists("Data/" + str(object_label)):
            os.mkdir("Data/" + str(object_label))
        cv2.imwrite("Data/" + str(object_label) + "/" + str("image") + str(i) + ".png", frame)

cap.release()
cv2.destroyAllWindows()