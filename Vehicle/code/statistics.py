__auther__ = "GuangweiXia"
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('qt4agg')
# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'

def statistics(path,cls):
    x = [i[0] for i in enumerate(cls)]
    data = []
    count = 0
    ratio = []
    mark = []
    for i in cls:
        data.append(len(os.listdir(path + '/' + i)))
        count = count + len(os.listdir(path + '/' + i))
    for i in data:
        ratio.append(i/count)
    for i in cls:
        mark.append(i)
    # 测试
    print('所对应标签' + str(mark))
    print('所对应数量' + str(data))
    print('各标签所占比例' + str(ratio))
    print('数据总数' + str(count))
    # 打印直方图
    plt.figure(figsize=(20, 10))
    plt.xlabel('labes')
    plt.ylabel('count')
    plt.title("Quantity distribution diagram")
    plt.bar(x, data)
    plt.xticks(x, mark, rotation=0)
    plt.show()
    # 打印饼状图
    plt.figure(figsize=(10, 10))
    plt.axes(aspect=1)
    plt.pie(x=data, labels=mark, autopct='%.0f%%', pctdistance=0.7)
    plt.title("Proportional distribution diagram")
    plt.show()


path1 = 'Data/字母'
path2 = 'Data/数字'
path3 = 'Data/汉字'
cls1 = os.listdir(path1)
statistics(path1,cls1)

cls2 = os.listdir(path2)
statistics(path2,cls2)

cls3 = os.listdir(path3)
statistics(path3,cls3)
