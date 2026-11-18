import numpy as np
import pandas as pd
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T


def outlier(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25, 50, 75]) # 25% 지점, 50%지점, 75% 지점을 찾는다
    # print('1사분위 : ', quartile_1) # 25% 지점
    # print('q2 : ', q2) # 중위수
    # print('3사분위 : ', quartile_3) # 3사분위 수 / 75% 지점
    # IQR(Interquartile Range, 사분위수 범위)
    iqr = quartile_3 - quartile_1
    # print('IQR : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5) # 아래쪽으로 iqr * 1.5데이터까지는 이상치로 안보겠다
    upper_bound = quartile_3 + (iqr * 1.5) # 위쪽으로 iqr * 1.5배 데이터 까지는 이상치로 안보겠다

    # outlier_loc : 이상치 값들, IQR, lower_bound, upper_bound
    return np.where((data > upper_bound) | (data < lower_bound)), \
            iqr, lower_bound, upper_bound


# ==========================================
# #1 이상치를 찾아낸다 & #2 이상치를 nan 처리를 한다
# ==========================================
aaa = aaa.astype(float)
for i in range(aaa.shape[1]):
    outlier_loc, iqr, low, up = outlier(aaa[:, i])  #1 이상치 위치 탐색
    aaa[outlier_loc, i] = np.nan                    #2 이상치를 nan으로 직접 변경

    # np.clip은 배열의 값들이 지정한 최소값(low)과 최대값(up) 범위를 벗어나지 않도록 경계값으로 잘라내어 제한하는 넘파이 함수
    # aaa[:, i] = np.clip(aaa[:, i], low, up)         # 결측치 바로 처리 low, up 값으로 사용

# print(aaa)
# exit()

# ==========================================
# #3 nan을 처리
# ==========================================

# pd.DataFrame(aaa) : 판다스 데이터프레임으로 변환
# .interpolate(method='linear', limit_direction='both') : 결측치 보간
# 'linear' (선형 보간 - 기본값), 
# 'limit_direction' 기본값은 'forward'이며, 특히 맨 앞이나 맨 뒤에 NaN이 있을 때 이 옵션이 매우 중요하게 작동합니다.
# .to_numpy() : 다시 넘파이 배열로 복원
aaa = pd.DataFrame(aaa).interpolate(method='linear', limit_direction='both').to_numpy()

print("\n=== #3 최종 변경된 aaa (중앙값 대체) ===")
print(aaa)

