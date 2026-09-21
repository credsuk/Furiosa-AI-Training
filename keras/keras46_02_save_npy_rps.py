
import numpy as np
from keras.preprocessing.image import ImageDataGenerator



#1. 데이터

datagen = ImageDataGenerator(
    rescale=1./255,
)

path_data = "./_data/rps/"
data = datagen.flow_from_directory(
    path_data,
    target_size=(100,100),
    batch_size=8000,
    class_mode='categorical',
    color_mode='rgb',
    
)

print(data[0][0].shape, data[0][1].shape) # (2048, 100, 100, 3) (2048, 3)

path_np = "./_data/rps_npy/"
np.save(path_np + "rps_x_npy.npy", data[0][0])
np.save(path_np + "rps_y_npy.npy", data[0][1])





