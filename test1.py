import tensorflow as tf

# 验证gpu是否有效
# tf.test.is_gpu_available()

# tf.config.list_physical_devices('GPU')

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# with tf.device('/GPU:0'):
#     print("111")

msg = tf.constant('Hello, TensorFlow!')
tf.print(msg)

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.compat.v1.Session() as sess:
    print(sess.run(c))
