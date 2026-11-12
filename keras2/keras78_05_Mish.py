import numpy as np
import matplotlib.pyplot as plt

# -5 ~ 5 까지 0.1 사이로 있는 데이터
x = np.arange(-5, 5, 0.1)

mish = lambda x: x * np.tanh(np.log(1 + np.exp(x)))

y = mish(x) 

plt.plot(x, y)
plt.grid()
plt.show()

