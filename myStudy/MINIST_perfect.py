import tensorflow as tf  # tensorflow를 tf라는 이름으로 import함
from keras import datasets, layers, models  # 신경망 구성을 위해 keras를 import
import matplotlib.pyplot as plt   # 이미지를 보여주거나 그래프를 그리기 위한 모듈
import numpy as np       # 다양한 수학적 처리를 위해 import
(train_images, train_labels), (test_images, test_labels)=datasets.mnist.load_data()
# MNIST데이터 셋을 가지고와서 (학습 이미지, 정답), (테스트 이미지, 정답)으로 나누어 줌
train_images=train_images.reshape((60000, 28, 28, 1))  # 학습용
# 6만장을 batch size로 묶어 한거번에 처리, 28x28크기, channel은 1이므로 흑백 이미지
test_images=test_images.reshape((10000, 28, 28, 1))    # 테스트용
# 만장을 batch size로 묶어 한거번에 처리, 28x28크기, channel은 1이므로 흑백 이미지
train_images, test_images=train_images/255, test_images/255
# RGB값 0-255를 0과 1사이로 표현해야 되므로 255로 나누어 정규화를 해줌 
################ Feature Extraction <Convolution Block> #################
model=models.Sequential()  # 신경망 모델을 만들고, 신경망을 순차적으로 연결해줌
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
# 합성곱(convolution) 계층(layer)을 만들어 신경망에 붙여줌(add), 32는 노드개수
# (3,3)은 필터(마스크)사이즈, 활성함수는 reLU사용, 이미지의 사이즈는 28x28, 흑백(1) 
model.add(layers.MaxPooling2D((2,2)))
# 풀링 계층을 추가해줌. 풀링을 통한 출력 사이즈는 (2,2)
model.add(layers.Dropout(0.25))     # dropout을 통해 25%의 연결을 끊음
# 위의 과정을 여러번 반복, 신경망의 합성곱 계층의 노드 갯수를 바꾸어도 됨
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Dropout(0.25))     # 25%의 연결을 끊음  
################## Fully Connected NN <Neual Net Block> ###################
model.add(layers.Flatten())  # 최종 출력된 이미지 배열을 평탄화해 입력해줌
model.add(layers.Dense(64, activation='relu')) # 은닉층, 노드 개수는 64
model.add(layers.Dense(10, activation='softmax')) # 출력층, softmax를 써서 확률값으로 출력
##########################  <Optimization Block>  ##########################
# 최적화를 위해 adam을 사용. 손실함수로 cross entropy를 사용, 평가지표는 Accuacy(정확도)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit함수를 통해 훈련(이미지 학습)을 시킴, epoch는 일단 1회로 설정
model.fit(train_images, train_labels, epochs=1)   # epoch는 1번만 했음
##########################  Image Test #####################
img_no=463      # 테스트 이미지의 번호 (463은 464번째 이미지)
test_image=test_images[img_no, :, :, 0] # 이미지를 보기 위해서 shape을 (28x28)형태로 만들어 줌
plt.title("Number of the Image: {}".format(test_labels[img_no]))
plt.imshow(test_image)   # 테스트 이미지(여기서는 464번째 이미지)를 보여줌 
plt.show()   # 464번째 수가 6이라는 것과 정답이 6이라는 것을 확인할 수 있음
# 이미지가 표시된 plot의 오른쪽 상단 x를 눌러주면 터미널 창에 에측값을 보여줌
##########################  Prediction with Probability ######################
# 에측을 위해 (bach size, height, width, channel)형태로 바꾸어 줌
pred=model.predict(test_image.reshape(1, 28, 28, 1)) 
print(pred)   # 0-9까지의 확율이 list 형태로 표시됨, 7번째(숫자 6을 의미)가 확율이 높은 것을 확인
num=np.argmax(pred)  # Numpy모듈의 argmax함수를 통해 가장 높은 수를 가지고 옴
print("예측값: {}".format(num))  # 예측된 숫자를 표시해줌