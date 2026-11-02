import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
# KFold가 나오면 항상 나오는게 cross_val_score(교차검증)
from sklearn.tree import DecisionTreeClassifier


#1. 데이터
x, y = load_iris(return_X_y=True)
# print(x, y)

# KFold 준비
n_splist = 5
kfold = KFold(n_splits=n_splist, shuffle=True, random_state=333) # n_splits 만큼 나눠준다 / 지금은 5등분



#2. 모델구성
model = DecisionTreeClassifier()


#3. 컴파일, 훈련
# kFold만금 훈련시키고 5등분해서 전체 데이터는 매번 훈련시킨다
score = cross_val_score(model, x, y, cv=kfold) # cv : cross_valdation

#4. 평가, 예측
# 액큐러시스쿼로 확인한다
# 모든 분류모델은 acc다
# 모든 회귀모델은 r2

# np.mean(score)는 / 배열(array)의 산술평균을 계산하는 함수
print("acc : ", score, '\n평균 acc :"', round(np.mean(score), 2))
# acc :  [0.93333333 0.9        1.         0.96666667 0.9       ] 
# 평균 acc :" 0.94