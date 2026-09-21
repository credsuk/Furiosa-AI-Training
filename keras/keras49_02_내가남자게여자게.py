
from keras.models import load_model
import numpy as np

#1. 데이터

#2. 모델 구축

#3. 컴파일, 훈련

model_path = "./_save/keras49/"
model = load_model(model_path + "men_woman_save01_model.keras")

#4. 평가 예측

from tensorflow.keras.preprocessing.image import load_img, img_to_array
my_image = "./_data/image/myPhoto.jpg"
img = load_img(my_image, target_size=(100,100)) # 이미지 한장짜리 불러올때 편하다
arr = img_to_array(img) 
arr = np.expand_dims(arr, axis=0) # reshape랑 동일 차원을 확장하겠다, 0번째에 넣겠다


print("=================================")
arr = arr/255.0
y_predict = model.predict(arr)

pred_val = y_predict[0][0]

# 2. 조건문에 pred_val을 사용하고, 남자의 경우 (1 - pred_val)로 확률을 재계산합니다.
if pred_val > 0.5:
    result = f"{pred_val:.1%}" # 소수점 첫째 자리까지 표시 (예: 85.3%)
    print(result, " 확률로 여자")
else:
    result = f"{(1 - pred_val):.1%}" # 남자의 확률은 (1 - 여자확률)입니다.
    print(result, " 확률로 남자")



