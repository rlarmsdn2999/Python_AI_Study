# Mediapipe에서 제공하는 Hands함수를 통해 손의 21개 랜드마크를 획득
# HandTracking_LandMarkPosition.py를 개조하여 사용하기 쉽도록 모듈화하였음
# 랜드마크에 대해서는 Hand_Landmarks.png 참조

import cv2
import mediapipe as mp          # 손과 랜드마크 인식용
import time                     # 영상 프레임 속도 (FPS) 체크용

class handDetector():    # Hands클래스의 Property 참조할 것
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHand = mp.solutions.hands          # Mediapipe를 통한 손 검출
        self.hands = self.mpHand.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # 영상위에 그림을 그려주기 위한 모듈

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR을 RGB로 변환
        self.results = self.hands.process(imgRGB)  # RGB영상으로 바꾸고 hands모듈 수행

        if self.results.multi_hand_landmarks:    # 두개의 손의 랜드마크가 잡힌다면
            for handLms in self.results.multi_hand_landmarks:  # 랜드마크와 연결선을 표시
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHand.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape  # 전체 영상 이미지의 높이, 폭, 채널 획득
                cx, cy = int(lm.x*w), int(lm.y*h) # 영상에서 랜드마크의 위치점 획득
                # print(id, cx, cy)  # 랜드마크 번호, x위치, y위치 표시      
                lmList.append([id, cx, cy])     
                if draw:       
                # if id == 0:  # id 0번(손목)의 위치에 핑크색 원을 표시
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)       ###### 영상 획득 기본 코드 ###### 0:첫번째 캠

    detector = handDetector()  # handDetector class의 instance 생성
    while True:
        success, img = cap.read()   ###### 영상 획득 기본 코드 ######    
        img = detector.findHands(img)
        # img = detector.findHands(img, draw=False)  # 랜드마크 기본 마킹이 사라짐
        lmList = detector.findPosition(img)
        # lmList = detector.findPosition(img, draw=False) # 랜드마크 추가 마킹이 사라짐
        if len(lmList) != 0:
            print(lmList[4])    # 엄지 끝점(랜드마크 4번) 위치 표시 

        cTime = time.time()     # time함수를 써서 현재시간 체크
        while cTime != pTime:
            fps = 1/(cTime-pTime)   # 현재 프레임과 전 프레임의 시간간격으로 fps계산 
            pTime = cTime           # 현재시간을 전시간으로 설정

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)    ###### 영상 획득 기본 코드 ######
        if cv2.waitKey(1) & 0xFF ==ord('q'):     # 키보드의 q 키가 입력되면 정지
            break

if __name__ == "__main__":
    main()