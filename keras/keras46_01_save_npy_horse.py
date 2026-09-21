
import numpy as np
from keras.preprocessing.image import ImageDataGenerator



#1. 데이터

datagen = ImageDataGenerator(
    rescale=1./255,
)

path_data = "./_data/horse_or_human/"
data = datagen.flow_from_directory(
    path_data,
    target_size=(200,200),
    batch_size=2000,
    class_mode='binary',
    color_mode='rgb',
    
)

print(data[0][0].shape, data[0][1].shape) # (1027, 200, 200, 3) (1027,)

path_np = "./_data/horse_or_human_npy/"
np.save(path_np + "horse_or_human_x_npy.npy", data[0][0])
np.save(path_np + "horse_or_human_y_npy.npy", data[0][1])





