import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split # 메소드명 그대로 훈련, 테스트, 나눔

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

# 사이킷런 Scikit-learn 사용해서 훈련, 테스트 데이터 분리
# x, y를 넣었기 때문에 순서대로 리턴됨 (파이썬 짱인데?)
# 훈련과 테스트 두개중에 하나만 할수도 있음
x_train, x_test, y_train, y_test = train_test_split( 
    x, y,               # x, y 동일하게 데이터가 삭제되야 문제가 없으니 같이 넣어야됨 데이터는 3개 이상도 가능
#    train_size = 0.7,  # 훈련 데이터를 70%로 지정함 소수로 입력하면됨 
#    shuffle = True,    # 데이터를 섞는 기능 기본값이 True
#    test_size =  0.3,  # 테스트 사이즈를 지정해 줄 수 있다 기본값 0.25
    random_state=223,   # 난수표 / 랜덤을 추출하는 난수값을 고정한다, 일관성있는 데이터를 얻기 위해 설정 / 고정된 랜덤
) 

print("x_train : ", x_train)
print("x_test : ", x_test)
print("y_train : ", y_train)
print("y_test : ", y_test)


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(11))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=4)

print("===========================================================")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

# results = model.predict(np.array([11]))
results = model.predict(x)
print("results :", results)

# 시각화 / 그래프 그리기
import matplotlib.pyplot as plt

plt.scatter(x, y) # scatter는 점찍는 메소드
plt.plot(x, results, color='red') # 선을 긋는 메소드
plt.show() # 표시할때 사용