import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

while cap.isOpened():
    ret, frame = cap.read()



    cv.imshow('frame', frame)
    if cv.waitKey(1) == 27:
        break
