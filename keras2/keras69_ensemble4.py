# keras69_ensemble3 복사

import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T # (100, 2)

y1 = np.array(range(3001, 3101)) # (100, )
y2 = np.array(range(13001, 13101)) # (100, )


x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, y1, y2,
    random_state=213, 
    test_size=0.1
)


# print(x1_train.shape, x1_test.shape) # (90, 2) (10, 2)
# print(x2_train.shape, x2_test.shape) # (90, 3) (10, 3)
# print(y_train.shape, y_test.shape) # (90,) (10,)


#2-1. 모델 구성
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(80, activation='relu', name='han4')(dense3)


#2-4. 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate
merge2 = Dense(20, name='mg2')(output1)
merge3 = Dense(5, name='mg3')(output1)


# y값을 분기 / Output가 2개일때
#2-5. 분기 1
last_dense1 = Dense(10, name="ld1")(merge3)
last_dense2 = Dense(10, name="ld2")(last_dense1)
last_output1 = Dense(1, name="last1")(merge3)

#2-5. 분기 2
last_output2 = Dense(1, name="last2")(merge3)


model = Model(inputs=[input1], outputs=[last_output1, last_output2])

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()
model.fit([x1_train,], [y1_train, y2_train], epochs=300, batch_size=8)
end_time = time.time()


#4. 평가 예측
result = model.evaluate([x1_test,], [y1_test, y2_test])
print("loss :: ", result)

x1_pred = np.array([range(100,106), range(400,406)]).T
# x2_pred = np.array([range(200,206), range(510,516), range(249, 255)]).T
# x3_pred = np.array([range(100,106), range(400,406), range(177, 183), range(133,139)]).T


# 예측값 완성
predict = model.predict([x1_pred])
print("predict :: ", predict)



# loss ::  [0.00010836720321094617, 0.00010426640801597387, 4.100799742445815e-06]
# predict ::  [array([[3091.1218], [3092.121 ], [3093.1204], [3094.1199], [3095.1199], [3096.1187]], dtype=float32), 
            #  array([[13058.294],[13059.294],[13060.294],[13061.294],[13062.295],[13063.293]], dtype=float32)]