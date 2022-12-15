import cv2 as cv
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, MaxHands=2, complexity=1, detectionConf=0.5, trackConf=0.5):
        self.mode=mode
        self.MaxHands=MaxHands
        self.complexity=complexity
        self.detectionConf=detectionConf
        self.trackConf=trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.MaxHands, self.complexity, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, frame, draw=True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=False):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w = frame.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(frame, (cx, cy), 15, (255, 255, 0), cv.FILLED)

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = HandDetector()
    while cap.isOpened():
        ret, frame = cap.read()

        detector.findHands(frame, draw=False)
        lmList = detector.findPosition(frame)

        if len(lmList) !=0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(frame, "FPS "+str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == 27:
            break


if __name__ == '__main__':
    main()