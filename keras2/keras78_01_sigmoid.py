import numpy as np
import matplotlib.pyplot as plt # 그림 그릴꺼야

# 시그 모이드식
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 람다로 변경
sigmoid = lambda x:  1 / (1 + np.exp(-x))

# -5 ~ 5 까지 0.1 사이로 있는 데이터
x = np.arange(-5, 5, 0.1)

print(x)
print(len(x)) # 100

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()

