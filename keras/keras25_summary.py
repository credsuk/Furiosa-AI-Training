from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))
# 파라미터의 개념 위에건 총 30개 1*3 + 3*4 + 4*3 + 3*1  이렇게 배웠음
# 어떤 추촌모델을 가져오면 우선 파라미터가 몇개인지 확인하게 된다
model.summary() # 모델의 파라미터를 확인하는 명령어

# 확인해봤더니.. 41개가 나옴 / 이유가 뭘까?
# 노드 하나당 바이어스가 포함되서 그런다






#3. 컴파일, 훈련



#4. 평가 예측



