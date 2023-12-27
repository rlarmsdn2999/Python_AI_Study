# 이미지 저장 파일명 사용을 위한 datetime
# 이미지 저장을 위한 cv2.imwrite
# q를 누르면 종료, s를 누르면 이미지 저장

import cv2   # pip install opencv-python
import os    # 파일 경로를 위해 import
import datetime # 현재 날자, 시간 등을 읽어오기 위해 import  
from imutils.video import WebcamVideoStream  # Webcam영상 스트림을 위해 import
# py -m pip install imutils

# Webcam을 host로 하여 ip어드레스와 port:4747지정
# DroidCam App상에 표시된 IP주소와 포트번호를 입력 
host = "{}:4747/video".format("http://192.168.0.43") # 유동 IP주소이므로 매번 바뀔 수 있음
cam = WebcamVideoStream(src=host).start()    # 비디오 스트림 시작. capture = cv2.VideoCapture(0) 부분에 해당 

while True:    # q키 입력으로 영상 종료
    frame = cam.read()    # 웹캠 영상을 읽어와 실시간으로 뿌림. ret, frame = capture.read() 에 해당  
    cv2.imshow('Original Video', frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('s'):
        file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + '.jpg'
        path = r'D:\AI Study\OpenCV\photo' # s키를 눌러 찍은 이미지를 저장할 경로
        cv2.imwrite(os.path.join(path , file), frame) # 경로와 파일명을 합쳐서 저장
        # cv2.imwrite(file, frame)
        print(file, ' saved')

cv2.destroyAllWindows()