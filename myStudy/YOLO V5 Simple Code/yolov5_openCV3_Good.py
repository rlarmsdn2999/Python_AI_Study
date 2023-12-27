##################### 짧으면서 파워풀한 모델이니 잘 숙지 할 것 #####################
# model부분을 Local PC상에 Laod하여 사용. 인터넷 없이도 사용 가능!
# 모델이 들어있는 위치를 파악하여 그 경로를 넣어줄 것
# path='yolov5n.pt'을 atom.pt(atom을 검출하도록 학습시켜 놓았음)로 바꾸면 atom을 검출할 수 있음
# path='yolov5n.pt'을 mask.pt(마스크쓴 사람을 검출하도록 학습시켜 놓았음)로 바꾸면 마스크쓴 사람을 검출할 수 있음

import cv2
import torch

#model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
# 처음에는 윗 모델에 억세스하여 로컬 PC에 모델을 다운받고, 그 위치를 확인(실행 후, 터미널 창에서 확인 가능)후 path를 붙여 넣을 것
# Using cache found in C:\Users\office1/.cache\torch\hub\ultralytics_yolov5_master <- 터미널 창에서 확인
model = torch.hub.load('C:/Users/User/.cache/torch/hub/ultralytics_yolov5_master', 'custom', path='C:/Users/User/Desktop/rlarmsdn/python/YOLO V5 Simple Code/best.pt', source='local')

cam = cv2.VideoCapture(0)
zoom_rate = 2      # 640x480사이즈를 2배로 확대

while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (640*zoom_rate, 480*zoom_rate))  # 640 X 480 사이즈를 확대하였음
    results = model(frame)
    results.render()
    cv2.imshow('yolov5', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()