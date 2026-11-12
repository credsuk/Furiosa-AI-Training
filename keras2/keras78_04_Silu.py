# 2017년 구글에서 swish 발표 / Silu의 다른 이름
# 음수에 이 부분을 전부 0처리 했더니 쫌 찝찝해서 조금 넣어봤다
# 그랬더니 좋을때가 있다
import numpy as np
import matplotlib.pyplot as plt

# -5 ~ 5 까지 0.1 사이로 있는 데이터
x = np.arange(-5, 5, 0.1)

silu = lambda x: x * (1/ (1 + np.exp(-x))) 

y = silu(x) 

plt.plot(x, y)
plt.grid()
plt.show()

