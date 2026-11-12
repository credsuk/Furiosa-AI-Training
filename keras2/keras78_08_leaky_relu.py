import numpy as np
import matplotlib.pyplot as plt

# -5 ~ 5 까지 0.1 사이로 있는 데이터
x = np.arange(-5, 5, 0.1)

alpha = 0.01
# leaky_relu = lambda x, alpha: np.maximum(alpha * x, x)
leaky_relu = lambda x, alpha: np.where(x>0, x, alpha*x) # 이렇게도 가능하다
y = leaky_relu(x, alpha) 

plt.plot(x, y)
plt.grid()
plt.show()

