import numpy as np
import matplotlib.pyplot as plt

# -5 ~ 5 까지 0.1 사이로 있는 데이터
x = np.arange(-5, 5, 0.1)
y = np.tanh(x) # -1 ~ 1 까지 탄젠트

plt.plot(x, y)
plt.grid()
plt.show()

