import os
import numpy as np
import tensorflow as tf
from keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator

# 데이터 루트 디렉토리 설정 (이미지를 저장할 위치)
trainPath = r'C:\Users\User\Desktop\RPS Game\datasets\train'
inputShape = (50, 50, 3)
numClass = 4  

# Hyperparameter 설정
numEpoch = 20
batchSize = 10
learningRate = 0.001
dropoutRate = 0.3

# 이미지 읽기 함수
def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_image(gfile, dtype=tf.float32)
    return image

# 학습 데이터 생성기 설정
train_dataGenerator = ImageDataGenerator(
    rescale=1.0/255.
)
# 업데이트된 데이터 디렉토리에 맞게 데이터 생성기 업데이트
train_generator = train_dataGenerator.flow_from_directory(
    trainPath,
    target_size=inputShape[:2],
    batch_size=batchSize,
    color_mode='rgb',
    class_mode='sparse'  # 'categorical'에서 'sparse'로 변경
)

# 모델 생성 함수
def create_and_train_model(train_generator):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropoutRate))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropoutRate))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(dropoutRate))

    # Dense 레이어의 뉴런 수를 numClass로 설정
    model.add(layers.Dense(numClass, activation='softmax'))

    model.compile(optimizer=tf.optimizers.Adam(learningRate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # 'categorical_crossentropy'에서 'sparse_categorical_crossentropy'로 변경

    # 모델 학습
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=numEpoch,
    )

    # 학습된 모델을 파일로 저장
    model.save('my_model.h5')

# 모델 생성 및 학습
create_and_train_model(train_generator)
