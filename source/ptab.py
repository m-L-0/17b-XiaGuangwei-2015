import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab


#数据集对应的标签
Lab={0:'T-shirt',1:'pants',2:'overpull',3:'dress',4:'Coat',5:'sandal',6:'shirt',7:'sneaker',8:'package',9:'ankle boot'}

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["../tfrecords/train.tfrecords"])

#解析图片案例
_, example = reader.read(filename_queue)  
features = tf.parse_single_example(
    example,features={
        'image_raw': tf.FixedLenFeature([], tf.string),  
        'label': tf.FixedLenFeature([], tf.int64),  
    })
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32) 

#建立会话
with tf.Session() as sess:
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image, label = sess.run([images, labels])
    image=image.reshape(28,28)
    for i in range(200):
        #plt.imshow(image)
        plt.show()
        print(Lab[label])
        print(image)