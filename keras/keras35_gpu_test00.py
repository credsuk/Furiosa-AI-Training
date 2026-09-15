import tensorflow as tf

print(tf.__version__)


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus) # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# 이 텐서플로우 버젼에서는 GPU가 실험적인 버젼이였음
# 한마디로 GPU가 있다


if(gpus):
    print('GPU 있다!')
else:
    print('GPU 없다!')



