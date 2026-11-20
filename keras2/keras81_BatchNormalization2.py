# keras36_cnn3_mnist 복사
# 
import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import BatchNormalization, Activation # 상당히 강력하다



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

####### 스케일링 2 #######
# 이미지에서 많이 사용하는거 2번째
# -1 ~ 1 사이로 만든다
# 255/2의 값으로 중간값을 먼저 빼고 그 후에 나누기를 하면 
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# Y 데이터의 원핫
from sklearn.preprocessing import OneHotEncoder # 분류이기 때문에 OneHot를 해야함
ohe = OneHotEncoder(sparse_output=False) # 뭔가 혼동스러운 행열(혼동 행열)을 방지하기 위해서 sparse_output 사용함
y_train = y_train.reshape(-1, 1) # 수치를 알면 60000, 1 이렇게 해도 되지만 모르면 전체라는 의미에서 -1, 1 이렇게 하면된다
y_test = y_test.reshape(-1, 1) # 사실상 -1, 1 이게 디폴트

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)



#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation=None)) # 그냥 두면 리니어니깐 None로 한다
model.add(BatchNormalization())
model.add(Activation('relu')) # 이렇게 BT다음에 논문처럼 relu를 해준다

model.add(GlobalAveragePooling2D())

model.add(Dense(units=8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
# 다중 행열 이니깐 softmax
model.add(Dense(10, activation='softmax')) # (10, )

model.summary()

# exit()

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
    batch_size=228,
    verbose=1,
    validation_split=0.2,
    callbacks=[es]
)
end_time = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("걸린시간 : ", round(end_time - start_time, 2), "초")
print("================= model.evaluate ===========================")
print("loss : ", loss[0])
print("acc : ", loss[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test, y_predict)
print("acc_score :", acc_score)



# 걸린시간 :  124.68 초
# loss :  0.04402141645550728
# acc :  0.9904999732971191
# acc_score : 0.9905

# 걸린시간 :  286.5 초
# ================= model.evaluate ===========================
# loss :  0.1430472582578659
# acc :  0.9592000246047974
# 313/313 [==============================] - 0s 1ms/step
# acc_score : 0.9592


# 걸린시간 :  513.64 초
# ================= model.evaluate ===========================
# loss :  0.19309116899967194
# acc :  0.9441999793052673
# 313/313 [==============================] - 3s 9ms/step



