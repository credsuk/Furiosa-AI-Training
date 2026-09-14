from tensorflow.keras.models import Sequential, Model # 함수형 모델을 Model이라고 한다
from tensorflow.keras.layers import Dense, Dropout, Input
# Input 이라는 레이어가 준비되어 있음

#1. 데이터
# 지금은 모델 시간! 넘어가는걸로!

#2-1. 순차적 모델구성
model = Sequential() # 모델정의
model.add(Dense(10, input_shape=(3,)))
model.add(Dropout(0.2))
model.add(Dense(9))
model.add(Dropout(0.2)) 
model.add(Dense(1))

model.summary()

########################################################################
#2-2. 함수형 모델
# 순차적 모델을 함수형 모델로 변경
input1 = Input(shape=(3,)) # Input를 사용해서 input 레이어 함수를 지정한다
dense1 = Dense(10, name='ys1', activation='relu')(input1) # 히든레이어 생성 이름은 임의로 지은거
# 연결하기 # Dense를 만들고 이름은 ys1인데 위에 input1과 연결해라 뒤에 ()부분은 연결하는 상단 부분
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(9, name="ys2")(drop1)
drop2 = Dropout(0.2)(dense2)
output1 = Dense(1)(drop2)
model2 = Model(inputs=input1, outputs=output1) # Model은 어디서부터 어디까지야 정의 해야횜 

model2.summary()
# 함수형은 InputLayer에 대한 부분도 보인다
# 파라미터는 두개가 동일하다




#3. 컴파일, 훈련


#4. 평가 예측