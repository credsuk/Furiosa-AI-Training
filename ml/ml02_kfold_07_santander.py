# keras42_cnn07_santander.py 복사

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor 

#1. 데이터
path = "c:/study/_data/kaggle_santander/"

train_csv = pd.read_csv(path + "train.csv", index_col=0,)
test_csv = pd.read_csv(path + "test.csv", index_col=0,)
submission = pd.read_csv(path + "sample_submission.csv", index_col=0,)


x = train_csv.drop(["target"], axis=1, )
y = train_csv["target"]

# KFold 준비
n_splist = 5
kfold = KFold(n_splits=n_splist, shuffle=True, random_state=5132) # n_splits 만큼 나눠준다 / 지금은 5등분



#2. 모델구성
model = DecisionTreeRegressor()


#3. 컴파일, 훈련
# kFold만금 훈련시키고 5등분해서 전체 데이터는 매번 훈련시킨다
score = cross_val_score(model, x, y, cv=kfold) # cv : cross_valdation
                                                                                                   
#4. 평가, 예측
print("acc : ", score, '\n평균 acc :"', round(np.mean(score), 2))
# acc :  [0.3449067  0.31596121 0.32422856 0.31675108 0.34921852] 
# 평균 acc :" 0.33

