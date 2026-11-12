import numpy as np
import matplotlib.pyplot as plt

# -5 ~ 5 까지 0.1 사이로 있는 데이터
x = np.arange(-5, 5, 0.1)

# SELU 고정 상수 (논문 제안값)
alpha = 1.6732632423543772
lmbda = 1.0507009873554805 # lambda인데 아래 함수가 있으니 한글자 삭제 
selu = lambda x, alpha, lmbda:lmbda * ((x>0) * x + (x<0) * (alpha * (np.exp(x) -1)))
y = selu(x, alpha, lmbda) 

plt.plot(x, y)
plt.grid()
plt.show()

