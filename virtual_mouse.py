import os
import cv2
import handtrackingModule as htm
import time
import numpy as np


brushThickness = 14
eraserThickness = 100
xp = 0
yp = 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

overlayList = []
folderPath = "header"
myList = os.listdir(folderPath)
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')

    overlayList.append(image)
header = overlayList[0]
drawColor = (255, 0, 0)


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.65)
while True:
    # phase 1:import the image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # phase 2: find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # phase 3: check which finger is up
    if len(lmList) != 0:
        print(lmList)
        x1, y1 = lmList[8][1:]  # tip of index fingers
        x2, y2 = lmList[12][1:]  # tip of middle fingers
        fingers = detector.fingersUp()
        # print(fingers)
    # phase 4: if selection mode: two finger are up!
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25),
                          drawColor, cv2.FILLED)
            # checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)

                if 850 < x1 < 1050:
                    header = overlayList[2]

                    drawColor = (0, 255, 0)
                if 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

            print("Selection Mode")

    # phase 5: if drawing mode: index finger is up!
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):

                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1),
                         drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1),
                         drawColor, brushThickness)
            xp, yp = x1, y1
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    # phase 6:setting the header image
    img[0:125, 0:1280] = header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    # cv2.imshow("Image inverse", imgInv)
    # cv2.imshow("Image Canvas", imgCanvas)
    cv2.waitKey(1)
