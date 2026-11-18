import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])

# 괜찮은 알고리즘 이다
def outlier(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25, 50, 75]) # 25% 지점, 50%지점, 75% 지점을 찾는다
    print('1사분위 : ', quartile_1) # 25% 지점
    print('q2 : ', q2) # 중위수
    print('3사분위 : ', quartile_3) # 3사분위 수 / 75% 지점
    # IQR(Interquartile Range, 사분위수 범위)
    iqr = quartile_3 - quartile_1
    print('IQR : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5) # 아래쪽으로 iqr * 1.5데이터까지는 이상치로 안보겠다
    upper_bound = quartile_3 + (iqr * 1.5) # 위쪽으로 iqr * 1.5배 데이터 까지는 이상치로 안보겠다
    return np.where((data > upper_bound) | (data < lower_bound)), \
            iqr, lower_bound, upper_bound

outlier_loc, iqr, low, up = outlier(aaa)
print('이상치의 위치 : ', outlier_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.axhline(up, color='pink', label="Upper Bound")
plt.axhline(low, color='pink', label="Lower Bound")
plt.legend()
plt.show()