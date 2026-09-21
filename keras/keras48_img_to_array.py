
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np

#1. 데이터
my_photo = "./_data/image/myPhoto.jpeg"
img = load_img(my_photo, target_size=(150,150)) # 이미지 한장짜리 불러올때 편하다

print(img) # <PIL.Image.Image image mode=RGB size=150x150 at 0x1E57F3DCF40>
print(type(img)) # <class 'PIL.Image.Image'>

# plt.imshow(img) # 확인할때 여기서 확인이 가능하다
# plt.show()

# 수치화를 시켜야 한다 
arr = img_to_array(img) 
print(arr)
print(arr.shape) # (150, 150, 3)
print(type(arr)) # <class 'numpy.ndarray'>

# arr = arr.reshape(1, 150, 150, 3)
arr = np.expand_dims(arr, axis=0) # reshape랑 동일 차원을 확장하겠다, 0번째에 넣겠다
print(arr.shape) # (1, 150, 150, 3)

np_path = "./_data/kaggle_cat_dog_npy/"
np.save(np_path + "keras48_my_image.npy", arr=arr)


