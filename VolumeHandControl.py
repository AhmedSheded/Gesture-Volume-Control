import cv2 as cv
import time
import numpy as np
import HandTrackingModule as hm

wCam, hCam = 640, 480
pTime = 0
cTime = 0

cap = cv.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)

detector = hm.HandDetector(detectionConf=0.7)

while cap.isOpened():
    ret, frame = cap.read()

    detector.findHands(frame, draw=True)
    points=detector.findPosition(frame)
    if len(points) >0:
        # print(points[0])
        x1, y1 = points[4][1], points[4][2]
        x2, y2 = points[8][1], points[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv.circle(frame, (x1, y1), 10, (50, 0, 255), cv.FILLED)
        cv.circle(frame, (x2, y2), 10, (50, 0, 255), cv.FILLED)
        cv.circle(frame, (cx, cy), 10, (50, 0, 255), cv.FILLED)
        cv.line(frame, (x1, y1), (x2, y2), (50, 0, 255), 3)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frame, "FPS: " + str(int(fps)), (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == 27:
        break