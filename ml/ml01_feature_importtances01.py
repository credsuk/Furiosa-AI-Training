from sklearn.datasets import load_iris
# 보통 뒤에 부분에  Classifier > 분류 / Regressor > 회기
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # 나무가 아니라 숲을 봐라
from sklearn.ensemble import GradientBoostingClassifier # 부스팅하면 조금 더 쎈놈이다

#  부스팅 중에서도 끝판왕은 아래! 아직 현역임 
from xgboost import XGBClassifier 

#1. 데이터
x, y = load_iris(return_X_y=True) # 바로 x, y 분리가 된다 / 실무에서는 사용할일 없다
# print(x.shape, y.shape) # (150, 4) (150,)

random_state = 12313
from sklearn.model_selection import train_test_split

# stratify=y 회기에서는 사용할 수 없다 분류에서만 사용한다
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, stratify=y)



#2. 모델구성
# 트리구조의 모델이고 아주 강력했다
model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
# XGBClassifier는 통상적으로 파라미터 튜닝을 해줘야 잘나온다
model4 = XGBClassifier(random_state=random_state)

models = [model1, model2, model3, model4]

#3. 컴파일, 훈련
# 모델 구성과 컴파일 없음
# model1.fit(x_train, y_train)
for model in models:
    model.fit(x_train, y_train)
    print("============================", model.__class__.__name__, "============================")

    #4. 평가, 예측
    score = model.score(x_test, y_test) # model.evaluate() 대신에 바로 나온다
    print("acc : ", score)

    # 리스레서(로스율) 스코어가 1.0이 나왔어 그럼 좋다고 나쁘다 할 수 없다 하지만 r2는 그렇게 나올 수 있다

    print(model.feature_importances_)
    # 확인해봤더니 3번째 컬럼이 가장 영향을 미치는 컬컴이다
    # F 스코어 라고 한다 / 높다고 무조건 좋은건 아니다
    # 나중에 F1 스코어 라고 있다 / 그건 알아야 한다
     

    











