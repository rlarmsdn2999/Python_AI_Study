# 모듈화된 포즈 트랙킹 파일을 이용하여 헬스 트레이너를 만들어 보자!
# https://www.youtube.com/watch?v=UY6-JzdnHUM  영상1:43 - 2:10

import cv2
import numpy as np
import time
import poseTrackingModule as pm

cap = cv2.VideoCapture(0)           ###### 영상 획득 기본 코드 ######
pTime = 0
detector = pm.poseDetector()
count = 0
dir = 0

while True:
    success, img = cap.read()       ###### 영상 획득 기본 코드 ######
    # img = cv2.imread("Resouce/test.jpg")   # 이미지 파일을 읽어와서 img로 표시
    img = detector.findPose(img, False)       # 포즈 랜드마크와 연결선 표시
    lmList = detector.getPosition(img, draw=False) # 랜드마크의 id별 x,y좌표 검출
    if len(lmList) != 0:
        # Right Arm
        angle = detector.getAngle(img, 12, 14, 16)    # 오른팔 어깨, 팔꿈치, 팔목간의 각도
        # # Left Arm
        # angle = detector.getAngle(img, 11, 13, 15)    # 왼팔 어깨, 팔꿈치, 팔목간의 각도
        per = np.interp(angle, (70,160), (0,100))   # 각도영역을 %로 변환 (각도는 미리 확인할 것) 
        # print(angle, per)

        # check for the dumbbell curls (아령 들기 횟수 체크)
        if per == 100:        # 팔을 완전히 폈을 때
            if dir == 0:      # 팔을 내리는 방향이라면 
                count += 0.5  # 내리고 올렸을 때 1회 이므로, 0.5회로 카운트
                dir = 1       # 팔을 올리는 방향으로 설정
        if per == 0:          # 팔을 완전히 접었을 때
            if dir == 1:      # 팔을 올리는 방향이라면
                count += 0.5  # 0.5회 추가, 올리고 내리면 1회로 카운트
                dir = 0       # 팔을 내리는 방향으로 설정

        # 온른쪽 부분에 팔의 움직임을 바 형태로 표시
        cv2.rectangle(img, (600,200), (640,500), (0,255,0), 3)
        cv2.rectangle(img, (600,int(per)*3+200), (640,500), (0,255,0), cv2.FILLED)
        cv2.putText(img, f'{100-int(per)}%', (550, 75), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

        # 화면 왼쪽 하단에 핑크색으로 아령들기 횟수 카운팅 표시
        cv2.rectangle(img, (0,400), (140,500), (0,255,0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (10, 475), cv2.FONT_HERSHEY_PLAIN, 6, (255,0,255), 10)


    # 아래쪽 4줄은 FPS연산 및 표시를 위한 코드
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow("Image", img)        ###### 영상 획득 기본 코드 ######
    if cv2.waitKey(1) & 0xFF ==ord('q'): # 키보드의 q 키가 입력되면 정지
        break