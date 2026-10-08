import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T # (100, 2)
                        # 삼성의 종가, 하이닉스 종가 / 라고 가정한다
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose() # (100, 3)
                        # 원유가                환율            금시세
y = np.array(range(3001, 3101)) # (100, )
                # 화성의 화씨 온도


x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, y, random_state=213, test_size=0.1)


# print(x1_train.shape, x1_test.shape) # (90, 2) (10, 2)
# print(x2_train.shape, x2_test.shape) # (90, 3) (10, 3)
# print(y_train.shape, y_test.shape) # (90,) (10,)


#2-1. 모델 구성
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(80, activation='relu', name='han4')(dense3)
# model1 = Model(inputs=input1, outputs=output1)


#2-2. 모델 구성
input21 = Input(shape=(3,))
dense21 = Dense(50, name='han21')(input21)
dense22 = Dense(40, name='han22')(dense21)
dense23 = Dense(30, name='han23')(dense22)
dense24 = Dense(20, name='han24')(dense23)
output21 = Dense(40, name='han25')(dense24)
# model2 = Model(inputs=input21, outputs=output21)


#2-3. 모델 합치기
# Concatenate 사슬 같이 잇다 라는 뜻
# 넘파이에서 했었다 동일한 메소드가 있다
# 그냥 이어 붙여지는 거다 연산이나 다른게 있는게 아님
from tensorflow.keras.layers import concatenate, Concatenate


# 앙상블 모델이  input이 모델 두개긴 하지만 전체적으로 보면 그냥 모델이다 그래서
# 대, 소문자는 문법적 차이로 다르다 내용은 똑 같다
# mearge1 = concatenate([output1, output21], name="mg1")
merge1 = Concatenate(name="mg1")([output1, output21])
merge2 = Dense(20, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)
last_output = Dense(1, name="last")(merge3)

model = Model(inputs=[input1, input21], outputs=last_output)

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()
model.fit([x1_train, x2_train], y_train, epochs=250, batch_size=8)
end_time = time.time()


#4. 평가 예측
result = model.evaluate([x1_test, x2_test], y_test)
print("loss :: ", result)

x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249, 255)]).T


# 예측값 완성
predict = model.predict([x1_pred, x2_pred])
print("predict :: ", predict)



# loss ::  0.0014938234817236662
# predict ::  [[3100.1714]
#  [3102.649 ]
#  [3105.1255]
#  [3107.6025]
#  [3110.0793]
#  [3112.556 ]]




