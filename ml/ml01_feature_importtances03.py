from sklearn.datasets import load_iris
# 보통 뒤에 부분에  Classifier > 분류 / Regressor > 회기
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier

#  부스팅 중에서도 끝판왕은 아래! 아직 현역임 
from xgboost import XGBClassifier 

#1. 데이터
x, y = load_iris(return_X_y=True) 
# print(x.shape, y.shape) # (150, 4) (150,)

random_state = 12313
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, stratify=y)

#2. 모델구성
model = XGBClassifier(random_state=random_state)


#3. 컴파일, 훈련
model.fit(x_train, y_train)
print("============================", model.__class__.__name__, "============================")

#4. 평가, 예측
score = model.score(x_test, y_test) # model.evaluate() 대신에 바로 나온다
print("acc : ", score) # # 프리퀀시 스코어
# acc :  0.9736842105263158

print("F 스코어 : ", model.feature_importances_)
# F 스코어 : [0.01679939 0.01034847 0.80158204 0.17127015]
# F 스코어 : 빈도수

# gain 스코어
gain_scores = model.get_booster().get_score(importance_type='gain') # 이거 중요하다
print("gain 스코어 : ", gain_scores) 
# gain 스코어 : {'f0': 0.05236059054732323, 'f1': 0.03225427493453026, 'f2': 2.498382806777954, 'f3': 0.5338173508644104}
# Gain 스코어 : 손실함수의 개선 점수. 높으면 개선 많이 됐다!
