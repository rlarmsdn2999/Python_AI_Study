import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480  # 웹캠 영상의 폭과 높이 픽셀수
cap = cv2.VideoCapture(1)         ###### 영상 획득 기본 코드 ###### 0:첫번째 캠

folderPath = "C:/Users/User/Desktop/rlarmsdn/python/Project 2023.8.28/RockPaperScissors AI Game/img"       # 이미지가 들어있는 폴더 경로 지정
myList = os.listdir(folderPath)   # folderPath안의 이미지를 myList에 저장

overlayList = []                  # 웹캠 영상위에 겹쳐서 표시하기 위한 이미지
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

totalFingers = 0  # 펼쳐진 손가락의 갯수, 0부터 5까지 테스트 해볼 것

while True:
    success, img = cap.read()       ###### 영상 획득 기본 코드 ######

    h, w, c = overlayList[totalFingers-2].shape  # 오버레이 이미지의 높이, 폭, 채널
    img[0:h, 0:w] = overlayList[totalFingers-2]  # 오버레이 이미지를 왼쪽 구석에 표시

    # cv2.rectangle(img, (20,225), (170,425), (0,255,0), cv2.FILLED) # 사각형안에 숫자 표시
    # cv2.putText(img, str(totalFingers), (45,375), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)

    cv2.imshow("Image", img)        ###### 영상 획득 기본 코드 ######
    if cv2.waitKey(1) & 0xFF ==ord('q'):     # 키보드의 q 키가 입력되면 정지
        break