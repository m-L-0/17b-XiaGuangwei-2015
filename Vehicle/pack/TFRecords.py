from PIL import Image
import os
import numpy as np
import tensorflow as tf

# 把传入的value转化为整数型的属性，int64_list对应着 tf.train.Example 的定义
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# 把传入的value转化为字符串型的属性，bytes_list对应着 tf.train.Example 的定义
def _bytes_feature(value):  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

images=[]
labels=[]
key_list=[]
value_list=[]
shape=[]
label_dict = {0: '云',1: '京',2: '冀',3: '吉',4: '宁',5: '川',6: '广',7: '新',8: '晋',9: '桂',10: '沪',11: '津',12: '浙',13: '渝',14: '湘',15: '琼',16: '甘',17: '皖',18: '粤',19: '苏',20: '蒙',21: '藏',22: '豫',23: '贵',24: '赣',25: '辽',26: '鄂',27: '闽',28: '陕',29: '青',30: '鲁',31: '黑'}
for key,value in label_dict.items():
    key_list.append(key)
    value_list.append(value)
nums = os.listdir('./Data/汉字/')
for i in range(len(nums)):
    img_temp=os.listdir('./Data/汉字/'+nums[i])
    for j in range(len(img_temp)):
        im=Image.open('./Data/汉字/'+nums[i]+'/'+img_temp[j])
        im=im.resize((24,48))
        labels.append(key_list[value_list.index(nums[i])])
        im = im.convert('L')
        im=np.array(im)
        images.append(im)
images=np.array(images)
labels=np.array(labels)
writer = tf.python_io.TFRecordWriter('./tfrecords/train.tfrecords')
for index in range(images.shape[0]): 
    #把图像矩阵转化为字符串  
    image_raw = images[index].tostring()  
    example = tf.train.Example(features=tf.train.Features(
        feature={
        'label': _int64_feature(labels[index]),  
        'image_raw': _bytes_feature(image_raw)}))  
    #将Example写入TFRecord文件
    writer.write(example.SerializeToString())   
writer.close()
print('汉字训练集已写入完毕,写入数量为',images.shape[0])

images=[]
labels=[]
key_list=[]
value_list=[]
shape=[]

for key,value in label_dict.items():  
    key_list.append(key)  
    value_list.append(value)

nums = os.listdir('./Data/汉字/')
for i in range(len(nums)):
    img_temp=os.listdir('./Data/汉字/'+nums[i])
    for j in range(len(img_temp)):
        im=Image.open('./Data/汉字/'+nums[i]+'/'+img_temp[j])
        im=im.resize((24,48))
        labels.append(key_list[value_list.index(nums[i])])
        im = im.convert('L')
        im=np.array(im)
        images.append(im)
images=np.array(images)
labels=np.array(labels)

writer = tf.python_io.TFRecordWriter('./tfrecords/test.tfrecords')
for index in range(images.shape[0]): 
    #把图像矩阵转化为字符串  
    image_raw = images[index].tostring()  
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(labels[index]),  
        'image_raw': _bytes_feature(image_raw)}))  
    #将 Example 写入TFRecord文件
    writer.write(example.SerializeToString())   
writer.close()
print('汉字测试集已写入完毕,写入数量为',images.shape[0])

