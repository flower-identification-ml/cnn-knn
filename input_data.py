import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -----------------生成图片路径和标签的List------------------------------------
train_dir = 'D:/ML/flower/input_data'

roses = []
label_roses = []
tulips = []
label_tulips = []
dandelion = []
label_dandelion = []
sunflowers = []
label_sunflowers = []

# step1：获取所有的图片路径名，存放到
# 对应的列表中，同时贴上标签，存放到label列表中。
def get_files(file_dir, ratio):
    for file in os.listdir(file_dir + '/roses'):
        roses.append(file_dir + '/roses' + '/' + file)
        label_roses.append(0)
    for file in os.listdir(file_dir + '/tulips'):
        tulips.append(file_dir + '/tulips' + '/' + file)
        label_tulips.append(1)
    for file in os.listdir(file_dir + '/dandelion'):
        dandelion.append(file_dir + '/dandelion' + '/' + file)
        label_dandelion.append(2)
    for file in os.listdir(file_dir + '/sunflowers'):
        sunflowers.append(file_dir + '/sunflowers' + '/' + file)
        label_sunflowers.append(3)
        # step2：对生成的图片路径和标签List做打乱处理
    image_list = np.hstack((roses, tulips, dandelion, sunflowers))
    label_list = np.hstack((label_roses, label_tulips, label_dandelion, label_sunflowers))

    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)


    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
        # 将所得List分为两部分，一部分用来训练tra，一部分用来测试val
    # ratio是测试集的比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))  # 测试样本数
    n_train = n_sample - n_val  # 训练样本数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels

