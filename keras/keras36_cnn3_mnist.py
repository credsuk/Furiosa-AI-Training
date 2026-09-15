# keras36_cnn2_mnist_inshow 복사
# 
import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)
print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test)) # 255 0
# 255가 나오면 문제없이 우리가 원하는 데로 나온거다


####### 스케일링 1 #######
# MinMax스케일링 원값에서 최소값을 뺀것을 최대값으로 나눈다
# 근데 우리는 이미 최소값이 0인걸 알고, 최대값이 255인걸 안다
# 그러면 MaxAbsScaler를 사용해도되고 최대값을 아니깐 바로 아래와 같이 해도된다
# 이미지라서 가능한거
# x_train = x_train/255. # .(플롯) 을 붙이면 플롯 형태로 간다
# x_test = x_test/255.

# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test)) # 1.0 0.0


####### 스케일링 2 #######
# 이미지에서 많이 사용하는거 2번째
# -1 ~ 1 사이로 만든다
# 255/2의 값으로 중간값을 먼저 빼고 그 후에 나누기를 하면 
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test))  # 1.0 -1.0







