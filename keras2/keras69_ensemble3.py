import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T # (100, 2)
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose() # (100, 3)
x3_datasets = np.array([range(100), range(301,401), range(77,177), range(33,133)]).T # (100, 4)

y1 = np.array(range(3001, 3101)) # (100, )
y2 = np.array(range(13001, 13101)) # (100, )


x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2,
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

#2-2. 모델 구성
input21 = Input(shape=(3,))
dense21 = Dense(50, name='han21')(input21)
dense22 = Dense(40, name='han22')(dense21)
dense23 = Dense(30, name='han23')(dense22)
dense24 = Dense(20, name='han24')(dense23)
output21 = Dense(40, name='han25')(dense24)

#2-3. 모델 구성
input31 = Input(shape=(4,))
dense31 = Dense(50, name='han31')(input31)
dense32 = Dense(40, name='han32')(dense31)
dense33 = Dense(30, name='han33')(dense32)
dense34 = Dense(20, name='han34')(dense33)
output31 = Dense(40, name='han35')(dense34)

#2-4. 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = Concatenate(name="mg1")([output1, output21, output31])
merge2 = Dense(20, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)


# y값을 분기 / Output가 2개일때
#2-5. 분기 1
last_dense1 = Dense(10, name="ld1")(merge3)
last_dense2 = Dense(10, name="ld2")(last_dense1)
last_output1 = Dense(1, name="last1")(merge3)

#2-5. 분기 2
last_output2 = Dense(1, name="last2")(merge3)


model = Model(inputs=[input1, input21, input31], outputs=[last_output1, last_output2])

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=300, batch_size=8)
end_time = time.time()


#4. 평가 예측
result = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print("loss :: ", result)

x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249, 255)]).T
x3_pred = np.array([range(100,106), range(400,406), range(177, 183), range(133,139)]).T


# 예측값 완성
predict = model.predict([x1_pred, x2_pred, x3_pred])
print("predict :: ", predict)



# loss ::  [0.019463133066892624, 0.004134535789489746, 0.015328598208725452]
# predict ::  [array([[3092.2637], [3093.2424], [3094.2214], [3095.2   ], [3096.179 ], [3097.156 ]], dtype=float32), 
            #  array([[13063.852], [13064.895], [13065.934], [13066.978], [13068.017], [13069.061]], dtype=float32)]



