import numpy as np
import matplotlib.pyplot as plt

# -5 ~ 5 까지 0.1 사이로 있는 데이터
x = np.arange(-5, 5, 0.1)

# 거짓 인쪽이 소멸됨 / x가 0보다 크면 * x | x가 0보다 작으면 (alpha*(np.exp(x)-1))
# alpha의 값만큼 마이너스 만큼로 내려간다
elu = lambda x, alpha: (x>0) * x + (x<0) * (alpha*(np.exp(x)-1))
y = elu(x, 2) 

plt.plot(x, y)
plt.grid()
plt.show()

