# keras42_cnn05_kaggle_bike.py 복사

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor 


#1. 데이터
path = "./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
# print(train_csv.shape, test_csv.shape, submission_csv.shape) # (10886, 11) (6493, 8) (6493, 2)

x = train_csv.drop(["casual", "registered", "count"], axis=1)
y = train_csv['count']
# print(x.shape, y.shape) # (10886, 8) (10886,)

# KFold 준비
n_splist = 5
kfold = KFold(n_splits=n_splist, shuffle=True, random_state=4213) # n_splits 만큼 나눠준다 / 지금은 5등분



#2. 모델구성
model = GradientBoostingRegressor()


#3. 컴파일, 훈련
# kFold만금 훈련시키고 5등분해서 전체 데이터는 매번 훈련시킨다
score = cross_val_score(model, x, y, cv=kfold) # cv : cross_valdation

#4. 평가, 예측
print("acc : ", score, '\n평균 acc :"', round(np.mean(score), 2))
# acc :  [0.33359152 0.31047136 0.33931798 0.33344597 0.33251733] 
# 평균 acc :" 0.33
