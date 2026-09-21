# keras48_img_to_array 복사

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

#1. 데이터
my_photo = "./_data/image/myPhoto.jpg"
img = load_img(my_photo, target_size=(150,150)) # 이미지 한장짜리 불러올때 편하다

# 수치화를 시켜야 한다 
arr = img_to_array(img) 
# print(arr.shape) # (150, 150, 3)

arr = np.expand_dims(arr, axis=0) # reshape랑 동일 차원을 확장하겠다, 0번째에 넣겠다
print(arr.shape) # (1, 150, 150, 3)

# np_path = "./_data/kaggle_cat_dog_npy/"
# np.save(np_path + "keras48_my_image.npy", arr=arr)

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

# 이번에는 디렉토리가 아니라 수치화된 데이터를 바로 쓸꺼야
# 수치화된 걸 사용할때는
# arr파일에 datagen이걸 적용 시키겠다
it = datagen.flow(
    arr, # 수치화된 데이터
    batch_size=1, # 실제 적으로 한번 했다
)

print(it) # <keras.preprocessing.image.NumpyArrayIterator object at 0x00000181F25F7E50>
print(it.next()) # 파이썬 3.10 까지 가능,
print(next(it)) # 파이썬 3.11부터는 문법이 이렇게 변했음
print(next(it).shape) # (1, 150, 150, 3)



# 그냥 플랏하면 한개만 보여줌
# 이제 서브 플랏은 여러게 보여줌
import matplotlib.pyplot as plt

fig, ax = plt.subplots(
    nrows=1, 
    ncols=5, #
    figsize=(5,5), # 5x5 사이즈 배경
)

for i in range(5):
    batch = next(it) # it를 5번 돌리겠다 라는뜻..! 
    # 한번돌리면  datagen.flow 가 실행되고  datagen.flow를 하기위해서 ImageDataGenerator를 호출한다 
    # 그래서 for가 5번 돌아서 ImageDataGenerator를을 5번 한 이미지데이터가 된다
    # 각각 다른데이터로 나오니깐 데이터가 늘어남
    print(batch.shape)
    batch = batch.reshape(150, 150, 3)

    ax[i].imshow(batch)
    # ax[i].axix('off') 축 필요 없음

plt.show()




