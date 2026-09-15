import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train)
# print(x_train[0])
# print(x_train[0][0])

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)
# 어라? 4차원이 아니네? 왜 그럴까? 실제 데이터는 (60000, 28, 28, 1)
# reshape 할때 순서는 바뀌지 않는다 값은 바꾸지 않는다

print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))
# y는 int형의 0부터 9까지 10개가 있음

print(pd.value_counts(y_test))
"""
1    1135
2    1032
7    1028
3    1010
9    1009
4     982
0     980
8     974
6     958
5     892
Name: count, dtype: int64
"""

import matplotlib.pyplot as plt

plt.imshow(x_train[5000], 'gray')
plt.show()


