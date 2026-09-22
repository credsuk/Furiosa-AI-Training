# keras50_flow2_next 복사

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)


################ 데이터 증폭 #####################
datagen = ImageDataGenerator(
    rescale=1./255,         # 파이썬은 "." 이 자료형이 붙으면 형변환이 된다 (실질적인 수치화는 이거 하나)
    width_shift_range=0.1,  # 평형 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range=6,       # 각도 조절(정해진 각도만큼 이미지 회전)
    zoom_range=0.2,         # 확대
    fill_mode='nearest',    # 채우는 함수 ( 이미지를 옴겼을때 비어져 있는 부분은 근처에 있는 데이터로 채워라)
)

augment_count = 40000 # 4만개

# 6만개중에 랜덤하게 4만개를 추출할꺼야
# randidx = np.random.randint(60000, size=augment_count) # int형으로 랜덤으로 추출한다

# x_train데이터는 변할 수 있으니 정수로 사이즈를 저렇게 적어주는거 보다 그냥 x_train의 크기를 입력할 수 있다
randidx = np.random.randint(x_train.shape[0], size=augment_count) # int형으로 랜덤으로 추출한다
print(randidx)
print(randidx.shape) # (40000,)
print(len(randidx)) # 40000 리스트와 튜블은 len으로 밖에 확인이 안된다
print(np.min(randidx), np.max(randidx)) # 1 / 59999 최소값 / 최대값


# 혹시 메모리 연동으로 인해 원 값이 변경될 수 있으니 .copy()로 확실하게 변경해준다
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,)

# 이렇게 사용하면 위에서 수정하면 아래쪽에서는 쭉~ 수정이 된다
x_augmented = x_augmented.reshape(
    x_augmented.shape[0], # 40000
    x_augmented.shape[1], # 28
    x_augmented.shape[2], # 28
    1
)

print(x_augmented.shape) # (40000, 28, 28, 1) 

x_augmented = datagen.flow(
    x_augmented, 
    y_augmented, 
    batch_size=augment_count,
    shuffle=False,

).next()[0] # x값만 필요하고 y값은 필요 없으니 첫번째 값만 뽑는다

#### 변환 완료 ###
print(x_augmented.shape)

x_train = x_train.reshape(60000, 28, 28, 1)
y_train = y_train.reshape(60000, )

# np.concatenate() 사슬처럼 역다 / 다차원도 같이 합쳐준다
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)
# print(np.unique(x_train, return_counts=True))



# MaxAbsScaler 데이터 스케일링
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

# OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.fit_transform(y_test.reshape(-1, 1))


#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (5,5), input_shape=(28, 28, 1)))
model.add(Dropout(0.2))
model.add(MaxPool2D())
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(Dropout(0.1))
model.add(Conv2D(62, (2,2), activation='relu'))
model.add(Dropout(0.3))
# model.add(Conv2D(41, (3,3), activation='relu'))
# model.add(MaxPool2D())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(44, activation='relu'))
model.add(Dense(10, activation='softmax')) # (10, )

model.summary() #  491,305



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True,
)


start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=5000,
    # epochs=1,
    batch_size=228,
    verbose=1,
    validation_split=0.2,
    callbacks=[es]
)
end_time = time.time()




#4. 평가 예측
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000, 10)


# loss = model.evaluate(x_test, y_test, verbose=1)
print("걸린시간 : ", round(end_time - start_time, 2), "초")
print("================= model.evaluate ===========================")
# print("loss : ", loss[0])
# print("acc : ", loss[1])



y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test, y_predict)
print("acc_score :", acc_score)



# 기존데이터
# 걸린시간 :  224.09 초
# loss :  0.26256242394447327
# acc :  0.9071999788284302
# acc_score : 0.9072

# 걸린시간 :  560.86 초
# acc_score : 0.9066