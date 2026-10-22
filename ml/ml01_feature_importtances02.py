from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

#  부스팅 중에서도 끝판왕은 아래! 아직 현역임 
from xgboost import XGBRegressor

#1. 데이터
x, y = load_diabetes(return_X_y=True) # 바로 x, y 분리가 된다 / 실무에서는 사용할일 없다
# print(x.shape, y.shape) # (150, 4) (150,)

random_state = 12313
from sklearn.model_selection import train_test_split

# stratify=y 회기에서는 사용할 수 없다 분류에서만 사용한다
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, )



#2. 모델구성
# 트리구조의 모델이고 아주 강력했다
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

models = [model1, model2, model3, model4]

#3. 컴파일, 훈련
# 모델 구성과 컴파일 없음
# model1.fit(x_train, y_train)
for model in models:
    model.fit(x_train, y_train)
    print("============================", model.__class__.__name__, "============================")

    #4. 평가, 예측
    score = model.score(x_test, y_test) # model.evaluate() 대신에 바로 나온다
    print("r2 : ", score)

    # print(model.feature_importances_)

    
