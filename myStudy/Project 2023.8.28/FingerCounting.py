# Mediapipe의 Hand모듈을 통해 손가락 숫자를 인식해 보자. 
# 앞서 작성한 HandTrackingModule.py를 import하여 사용
# 반드시 손을 세워서 영상을 획득해야 함

import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480  # 웹캠 영상의 폭과 높이 픽셀수
cap = cv2.VideoCapture(0)         ###### 영상 획득 기본 코드 ###### 0:첫번째 캠
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"       # 이미지가 들어있는 폴더 경로 지정
myList = os.listdir(folderPath)   # folderPath안의 이미지를 myList에 저장

overlayList = []                  # 웹캠 영상위에 겹쳐서 표시하기 위한 이미지
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIDs = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()       ###### 영상 획득 기본 코드 ######
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []
        
        # Thumb (엄지): 오른손일 경우는 if문의 부등호를 반대로 할 것 
        if lmList[tipIDs[0]][1] < lmList[tipIDs[0]-1][1]: 
            fingers.append(1)              # 엄지가 펴지면 1
        else:                                             
            fingers.append(0)              # 엄지가 접히면 0

        # 4 Fingers (검지, 중지, 약지, 소지)
        for id in range(1, 5):    # index, middle, ring, pinky
            if lmList[tipIDs[id]][2] < lmList[tipIDs[id]-2][2]:
                # print("Finger Opened")  # 손가락이 펴지면
                fingers.append(1)
            else:   # If finger is Closed # 손가락이 접히면
                fingers.append(0)

        # print(fingers) # 엄지부터 소지까지, 펴지면 1, 접히면 0으로 표시
        totalFingers = fingers.count(1)  # 펼쳐진 손가락의 갯수
        # print(totalFingers)

        h, w, c = overlayList[totalFingers-1].shape  # 오버레이 이미지의 높이, 폭, 채널
        img[0:h, 0:w] = overlayList[totalFingers-1]  # 오버레이 이미지를 왼쪽 구석에 표시

        cv2.rectangle(img, (20,225), (170,425), (0,255,0), cv2.FILLED) # 사각형안에 숫자 표시
        cv2.putText(img, str(totalFingers), (45,375), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)

    cTime = time.time()             # 현재 시간
    if cTime != pTime:
        fps = 1 / (cTime - pTime)       # fps 계산
        pTime = cTime                   # 현재시간을 전시간으로 세팅

    cv2.putText(img, f'FPS:{int(fps)}', (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow("Image", img)        ###### 영상 획득 기본 코드 ######
    if cv2.waitKey(1) & 0xFF ==ord('q'):     # 키보드의 q 키가 입력되면 정지
        break