import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터
import joblib
joblib_path = "./_save/joblib/"
df = joblib.load(joblib_path + "ml22_상관계수1.joblib")

print(df)

# 상관관계
print("==================== 상관관계 히트맵 ====================")
print(df.corr())
#                       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    Target
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941       0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126      -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865       0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000       0.956547
# Target                      0.782561         -0.426658           0.949035          0.956547       1.000000

import matplotlib.pyplot as plt
# pip install seaborn
import seaborn as sns # 그림을 좀 예쁘게 그려주는 놈
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()



