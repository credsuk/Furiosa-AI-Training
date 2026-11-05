import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=21332, test_size=0.2)

# 총 25번 돌린다
# 다돌려본다.. 그게 그리드서치
learnig_rate = [0.1, 0.05, 0.01, 0.005, 0.001] # 5번돌리고
max_depth = [3,4,5,6,7] # 5번 돌리고
best_score = 0
best_parameters = ""


for i, lr in enumerate(learnig_rate):
    for j, md in enumerate(max_depth): # 아주 중요한 파라미터 성능차이 많이남
        model = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_depth=md,
        )
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)

        if score > best_score :
            best_score = score
            best_parameters = lr, md
        print(i, ",", j, "번째")

print("최고 점수 : {:.2f}".format(best_parameters))





