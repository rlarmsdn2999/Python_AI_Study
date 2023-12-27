# runs폴더에 사진의 각 부분을 나누어 crop한 부분(person, tie..)을 확인해 보자!

import torch
model = torch.hub.load('C:/Users/User/.cache/torch/hub/ultralytics_yolov5_master', 'custom', path='C:/Users/User/Desktop/rlarmsdn/python/project/YOLO/bubbles/best(1).pt', source='local')
img = r'C:\Users\User\Desktop\rlarmsdn\python\project\YOLO\bubbles\datasets\img\train\image_11.jpg'
results = model(img)
results.crop(save = True)
results.show()