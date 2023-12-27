# MNIST처럼 폴더명이 Label로 되어 있을때 DataGenerator를 통해 간단히 로드하여 전처리를 할 수 있다.
# 이것이 최종본임으로 잘 숙지할 것
import os
from glob import glob
import numpy as np
import tensorflow as tf
from keras import datasets, layers, models
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

train_dir='C:/Users/User/Desktop/mnist/traning'
test_dir='C:/Users/User/Desktop/mnist/testing'

###################### Hyperparameter Tuning ####################
num_epoch=1                            # 훈련 에포크 회수
batch_size=32                          # 훈련용 이미지의 묶음
learning_rate=0.001                    # 학습률, 작을수록 학습 정확도 올라감
dropout_rate=0.3                       # 30%의 신경망 연결을 의도적으로 끊음. 과적합 방지용
input_shape=(50, 50, 1)                # 입력데이터(이미지)의 크기, 원하는 크기를 입력하면 모든 이미지가 resize됨
num_class=3                       # 분류를 위한 정답의 갯수

########################## Preprocess ############################
train_datagen=ImageDataGenerator(         # Datagenerator로 이미지를 변환시킴
    rescale=1./255.,                      # Normalize를 위해 255로 나누어 줌
    width_shift_range=0.3,                # 폭(가로) 쪽으로 30%범위에서 랜덤하게 좌우 시프트 시킴
    zoom_range=0.2,                       # 20%범위에서 랜덤하게 크기를 늘리거나 줄임
    horizontal_flip=True                  # 수평축을 중심으로 이미지를 뒤집음
)
test_datagen=ImageDataGenerator(          # 테스트 부분은 이미지 전환이 필요없으나, 스케일은 맞추어 주어야 함
    rescale=1./255.                       # train의 DataGenerator와 같은 크기로 rescale해야 함
)
train_generator=train_datagen.flow_from_directory(       # DataGenerator를 통해 데이터를 Load할 수도 있다
    train_dir,
    target_size=input_shape[:2],    # (28, 28, 1)에서 맨뒤의 1은 channel이므로 빼주어야 함. 즉, 앞의 2개만 가지고 옴
    batch_size=batch_size,          # 메모리 관리를 위해 훈련 데이터셋을 적당한 크기(장수)로 묶어주어야 함
    color_mode='grayscale',         # 칼라인 경우 'rgb' 또는 'rgba'로 설정해야 함
    class_mode='categorical'        # 현재는 카테고리를 찾아내는 모델이므로..., 2진법적 출력의 경우 Binary로 설정
)                                   # Found 60000 images belonging to 10 classes.라 표시됨을 확인
validation_generator=test_datagen.flow_from_directory(       # DataGenerator를 통해 데이터를 Load할 수도 있다
    test_dir,
    target_size=input_shape[:2],    # (28, 28, 1)에서 맨뒤의 1은 channel이므로 빼주어야 함. 즉, 앞의 2개만 가지고 옴
    batch_size=batch_size,          # 고해상도 사진은 작게 설정하자. 4나 8정도...
    color_mode='grayscale',
    class_mode='categorical'        # 현재는 카테고리를 찾아내는 모델이므로..., 2진법적 출력의 경우 Binary로 설정
)                                   # Found 10000 images belonging to 10 classes.라 표시됨을 확인

############### Feature Extraction <Convolution Block> ##############
model=models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))  
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(dropout_rate)) 
model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(dropout_rate)) 

################ Fully Connected NN <Neual Net Block> ################
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(dropout_rate)) 
model.add(layers.Dense(num_class, activation='softmax'))

########################  Optimization Block  ########################
model.compile(optimizer=tf.optimizers.Adam(learning_rate), 
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

###########################  Training Block  ##########################
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epoch,
    validation_data=validation_generator,      # Label된 testset이 없는 경우 생략하는 것이 맞음
    validation_steps=len(validation_generator) # Label된 testset이 없는 경우 생략하는 것이 맞음
)  
# val_loss는 검증 손실값으로 훈련도중 증가하면 Overfitting(과적합)이 일어난 것임
# 과적합이 일어나는 지점부터는 훈련의 의미가 없음. val_accuracy는 검증 정확도

########################  Prediction with Probability ####################
data_path=glob('C:/Users/User/Desktop/mnist/test')    # 전체 경로 지정
def read_image(path):               # 이미지를 읽어오기 위한 함수 만듦
    gfile=tf.io.read_file(path, 'rb')     # 경로상의 하나의 이미지를 읽어 들여 gfile변수에 보관
    image=tf.io.decode_image(gfile, dtype=tf.float32)  # 읽어들인 이미지를 디코딩하여 이미지 배열로 만듦
    return image

for test_no in range(3):    # 테스트 하고자 하는 이미지의 번호, class수가 10개 이므로 9이상은 9로 판단
    path=data_path[test_no-1]     # test_no+1번째 파일의 경로
    
    img=read_image(path)
    img=tf.image.resize(img, input_shape[:2])  # 이미지의 크기를 바꾸어 입력 이미지 크기와 동일하게 맞추어 줌 

    image=np.array(img)   # 저장된 이미지를 배열형태로 만듦(중요). 이러면 imshow를 통해 이미지를 볼 수 있음
    # print(image.shape)          # 배열형태로 만들어 졌기 떄문에 형태확인이 가능. shape이 28X28임을 확인
    plt.imshow(image, 'gray')
    plt.title('Check the Image and Predict Together!')
    plt.show()

    image=image[:, :, 0]        # 그림판을 통해 낙서를 하였기 떄문에 (28, 28)이미지가 (28, 28, 4)로 바뀌었음-> 마지막 4를 없애야 함
    test_image=image[tf.newaxis, ..., tf.newaxis]   # 테스트 이미지 shape을 (28, 28) -> (1, 28, 28, 1) 로 만들어 줌
    pred=model.predict(test_image)               # predict함수를 사용하여 테스트 이미지 값을 유추
    print(pred)                           # test_no번째의 이미지를 예측하여 확률분포로 보여줌
    num=np.argmax(pred)                   # 확률분포중 가장 큰 값을 찾아 숫자로 표시 
    print("예측값: {}".format(num))