
# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

import numpy as np
from keras.preprocessing.image import ImageDataGenerator



#1. 데이터

datagen = ImageDataGenerator(
    rescale=1./255,
)

path_data = "./_data/image/men_women/"
data = datagen.flow_from_directory(
    path_data,
    target_size=(100,100),
    batch_size=9000,
    class_mode='binary',
    color_mode='rgb',
)

path_np = "./_data/kaggle_men_women/"
np.save(path_np + "keras46_03_x_train.npy", data[0][0])
np.save(path_np + "keras46_03_y_train.npy", data[0][1])





