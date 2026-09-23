
# keras27_Scaler01_californaia.py 복사 해옴

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# 스케일링을 한다
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) # 카이킥 런에서 # 핏은 실행하다 라는 뜻 아직 실행한거 아님 준비 까지
x = scaler.transform(x) # transform은 이걸 실행하는거

print("shape ::", x.shape, y.shape) # (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=253)


#2. 모델구성
model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(8,)))
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# ReduceLROnPlateau / Reducel(조절하다), LR > Learning Rate, On, Plateau

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=20,
    verbose=1,
    factor=0.5 # 나누는 기준 기본값은 0.1
)

es = EarlyStopping(
    monitor='var_loss',
    mode='min',
    patience=50,
    restore_best_weights=True,
    verbose=1,
)



start_time = time.time()
hist = model.fit(
    x_train, y_train, 
    epochs=5000, 
    batch_size=32, 
    validation_split=0.2,
    callbacks=[es, rlr],
)
end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("========================================================")
print("훈련에 걸린시간 : ", round(end_time - start_time, 2), "초")
print("loss : ", loss)



# 훈련에 걸린시간 :  198.05 초
# loss :  0.326321005821228

# learning_rate 0.01
# 훈련에 걸린시간 :  185.51 초
# loss :  0.3349727988243103


# 훈련에 걸린시간 :  1872.19 초
# loss :  0.33117613196372986