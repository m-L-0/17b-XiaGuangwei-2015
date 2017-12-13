import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#入口函数
def main(FileName,Num):
    writer = tf.python_io.TFRecordWriter(FileName)
    for index in range(Num):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature=
            {
                'pixels': _int64_feature(pixels),
                'label': _int64_feature(np.argmax(labels[index])),
                'image_raw': _bytes_feature(image_raw)
            }))
        writer.write(example.SerializeToString())
    writer.close()

data_dir = "../fashion"
trainFile = "../tfrecords/train.tfrecords"
testFile = "../tfrecords/test.tfrecords"
mnist = input_data.read_data_sets(data_dir, dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[0]

#训练集和训练集数量
trainNum = mnist.train.num_examples
testNum = mnist.test.num_examples

main(trainFile,trainNum)
main(testFile,testNum)
print('训练集:',trainNum)
print('测试集:',testNum)