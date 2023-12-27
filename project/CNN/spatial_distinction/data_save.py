import os
import cv2
from imutils.video import WebcamVideoStream  # Webcam영상 스트림을 위해 import
# py -m pip install imutils

# 저장할 루트 디렉토리 경로 설정
save_root_dir = r'C:\Users\User\Desktop\RPS Game\datasets\test'

data1_Num = 0
data2_Num = 0
data3_Num = 0
data4_Num = 0

# 폴더를 만들기 위한 디렉토리 경로 설정
classroom = os.path.join(save_root_dir, '1')
restarea = os.path.join(save_root_dir, '2')
maindoor = os.path.join(save_root_dir, '3')
stair = os.path.join(save_root_dir, '4')

# 폴더가 없다면 생성
os.makedirs(classroom, exist_ok=True)
os.makedirs(restarea, exist_ok=True)
os.makedirs(maindoor, exist_ok=True)
os.makedirs(stair, exist_ok=True)

# Webcam을 host로 하여 ip어드레스와 port:4747지정
# DroidCam App상에 표시된 IP주소와 포트번호를 입력 
host = "{}:4747/video".format("http://192.168.0.43") # 유동 IP주소이므로 매번 바뀔 수 있음
capture = WebcamVideoStream(src=host).start()    # 비디오 스트림 시작. capture = cv2.VideoCapture(0) 부분에 해당 

while True:
    ret, frame = capture.read()
    if ret == True:
        cv2.imshow("VideoFrame", frame)

        keyinput = cv2.waitKey(33)

        if keyinput == ord('q'):
            break

        elif keyinput == ord('1'):
            data1_Num += 1
            save_path = os.path.join(classroom, f'{data1_Num}.jpg')
            save = cv2.imwrite(save_path, frame, params=None)
            print(f'1키가 눌렸습니다. 이 영상을 [교실]에 저장합니다({data1_Num}).')

        elif keyinput == ord('2'):
            data2_Num += 1
            save_path = os.path.join(restarea, f'{data2_Num}.jpg')
            save = cv2.imwrite(save_path, frame, params=None)
            print(f'2키가 눌렸습니다. 이 영상을 [쉼터]에 저장합니다({data2_Num}).')

        elif keyinput == ord('3'):
            data3_Num += 1
            save_path = os.path.join(maindoor, f'{data3_Num}.jpg')
            save = cv2.imwrite(save_path, frame, params=None)
            print(f'3키가 눌렸습니다. 이 영상을 [현관]에 저장합니다({data3_Num}).')
        elif keyinput == ord('4'):
            data3_Num += 1
            save_path = os.path.join(stair, f'{data3_Num}.jpg')
            save = cv2.imwrite(save_path, frame, params=None)
            print(f'4키가 눌렸습니다. 이 영상을 [계단]에 저장합니다({data3_Num}).')
    else:
        break

capture.release()
cv2.destroyAllWindows()
