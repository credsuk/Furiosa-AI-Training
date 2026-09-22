# keras51_flow1_next 복사

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)


################ 데이터 증폭 #####################

datagen = ImageDataGenerator(
    rescale=1./255,         # 파이썬은 "." 이 자료형이 붙으면 형변환이 된다 (실질적인 수치화는 이거 하나)
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 평형 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range=6,       # 각도 조절(정해진 각도만큼 이미지 회전)
    zoom_range=0.2,         # 확대
    # shear_range=0.7,        # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 짜부)
    fill_mode='nearest',    # 채우는 함수 ( 이미지를 옴겼을때 비어져 있는 부분은 근처에 있는 데이터로 채워라)
)

augment_count = 100
print(x_train[0].shape) # (28, 28)

# np.tile()  넘파이(NumPy)에서 입력한 배열 전체를 지정한 횟수만큼 반복하여 새로운 배열을 만드는 함수
# 이렇게 하면 100개를 만든다
# 증폭은 동일한 데이터로 만드는게 아니라 변환해서 붙여야 한다
# aaa = np.tile(x_train[0], augment_count)
# print(aaa.shape) # (28, 2800) 컬럼쪽으로만 쭉~ 이어 붙어졌다
# 우리가 원하는건 28짜리가 100기가 있어야 한다


aaa = np.tile(x_train[0], augment_count).reshape(-1, 28, 28, 1) # 그냥 여기서 바로 reshape()
print(aaa.shape) # (100, 28, 28, 1)


##### 증폭의 단순 복북!! ######
# flow에는 데이터 한개도 되지만 2개도 넣을 수 있음
xy_data = datagen.flow(
    # x값
    # 이렇게 하면 x_train에 데이터를 뽑아서 100, 28, 28, 1 으로 만든다
    # 1. 1차원 데이터로 우선 변경하고, x_train[0].reshape(28*28) = 784
    # 2. np.tile()로 복사해서 785*100 만듬
    # 3. 변경된 데이터를 reshape로 4차원으로 다시 변경 (100, 28, 28, 1) 
    np.tile(x_train[0].reshape(28*28), augment_count).reshape(-1, 28, 28, 1),

    # y값
    # zeros = 0 채워 넣어라 그걸 100개 으로 데이터 만든다
    np.zeros(augment_count),

    # 크기는 augment_count(100)
    batch_size=augment_count,

    # 섞지 않는다
    shuffle=False, 

# Iterator 라서 next
).next()

# print(xy_data)
# print(type(xy_data)) # <class 'tuple'>
# print(xy_data.shape) # AttributeError: 'tuple' object has no attribute 'shape'

print(len(xy_data)) # tuple 데이터는 len() 으로 확인한다 / 2 x와 y두개의 값이 나옴

print(xy_data[0].shape) # (100, 28, 28, 1)
print(xy_data[1].shape) # (100,)

fig = plt.subplots(
    figsize=(7,7), # 7x7 배경 사이즈
)

# 100장을 출력할 수 있지만 우리는 50장만 확인
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][i], cmap='gray')

plt.show()



