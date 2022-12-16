import cv2 as cv
import time
import numpy as np
import HandTrackingModule as hm
import math
import alsaaudio


volume = alsaaudio.Mixer()

wCam, hCam = 640, 480
pTime = 0
cTime = 0

cap = cv.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)

detector = hm.HandDetector(detectionConf=0.7)



def pixelInCm(x1, y1, x2, y2):
    dx = x1-x2
    dy = y1-y2
    pixels = np.sqrt(dx*dx+dy*dy)
    pixel = pixels/10
    return pixel


while cap.isOpened():
    ret, frame = cap.read()

    detector.findHands(frame, draw=True)
    points=detector.findPosition(frame)
    if len(points) >0:
        x1, y1 = points[4][1], points[4][2]
        x2, y2 = points[8][1], points[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        color = (255, 0, 255)

        pixel = pixelInCm(points[0][1], points[0][2], points[5][1], points[5][2])
        length = math.hypot(x2 - x1, y2 - y1) // pixel

        vol = np.interp(length, [0, 14], [-7, 100])
        if points[12][1] > points[9][1]:
            cv.circle(frame, (x1, y1), 10, color, cv.FILLED)
            cv.circle(frame, (x2, y2), 10, color, cv.FILLED)
            cv.circle(frame, (cx, cy), 10, color, cv.FILLED)
            cv.line(frame, (x1, y1), (x2, y2), color, 3)

            if vol<0:
                volume.setvolume(0)

            if vol >=0 and vol <101:
                volume.setvolume(int(vol))


        cv.putText(frame, "destance: " + str(length)+' cm', (300, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frame, "FPS: " + str(int(fps)), (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == 27:
        break