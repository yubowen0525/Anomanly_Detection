# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     preprocess
   Description :
   Author :       ybw
   date：          2020/11/16
-------------------------------------------------
   Change Activity:
                   2020/11/16:
-------------------------------------------------
"""
import os
import time
from collections import Counter
from copy import deepcopy

import numpy
import pandas as pd
import seaborn
from sklearn.model_selection import train_test_split

from utils.file import getFile, mkdir
from utils.sklearn import random, standardScaler, normalize, minMaxScaler, maxAbsScaler
from multiprocessing import Pool
from config.setting import dataset


def merge(filepath):
    return pd.read_csv(filepath)


def classification(directoryPath="Classification", standardizedWay="standard"):
    """
    Classification data set and standardized
    :param directoryPath: save path
    :param standardizedWay: standardized way
    :return: None
    """
    labels = {}
    files, _ = getFile(dataset + "\\MachineLearningCVE")
    pool = Pool(10)
    result = pool.map(merge, files)
    workPath = os.path.join(dataset, directoryPath)
    mkdir(workPath)
    for df in result:
        grouped = df.groupby(df[" Label"])
        for name, group in grouped:
            print(name, group.shape)
            group = group.drop([" Label"], axis=1)
            path = os.path.join(workPath, name + ".csv")
            x = group.values

            # 混洗
            numpy.random.shuffle(x)

            # 异常值处理
            x_type = numpy.nan_to_num(x, posinf=9999, neginf=-9999)

            # 标准化
            # if standardizedWay == "standard":
            #     x_scale = Standard(x_type)
            # else:
            #     x_scale = Normalized(x_type)

            data = pd.DataFrame(x_type)

            print(name, data.shape)
            if name not in labels.keys():
                labels[name] = data.shape[0]
            else:
                labels[name] += data.shape[0]
            data.to_csv(path, mode='a', encoding='utf-8', header=False, index=False)
    # path = os.path.join(dataset, "labels.csv")
    print(labels)
    # labels = pd.DataFrame.from_dict(labels, orient='index')
    # labels.to_csv(path, encoding='utf-8', index=False)


def constructDataSet(srcPath="Classification", savePath="DeepLearningDateSet", quantify="", supervised=True):
    files, _ = getFile(os.path.join(dataset, srcPath))
    savePath = os.path.join(dataset, savePath)
    mkdir(savePath)
    X = None
    Y = None
    Labels = []
    # 需要第一个文件一定是BENIGN
    for index, file in enumerate(files):
        labelName = os.path.basename(file).split(".")[0]
        Labels.append(labelName)
        x = pd.read_csv(file).values
        x = numpy.concatenate((x, x[:, -2:]), axis=1)
        n, m = x.shape
        if labelName == "BENIGN":
            if index != 0:
                print("BENIGN must be first file")
                exit(1)
            y = numpy.linspace(index, index, n)
            splitAbnormal = n
            print(splitAbnormal)
        else:
            y = numpy.linspace(index, index, n)

        if X is None:
            X = x
            Y = y
            continue
        print("X.shape:", X.shape, "x.shape:", x.shape)
        print("Y.shape:", Y.shape, "y.shape:", y.shape)
        X = numpy.concatenate((X, x), axis=0)
        Y = numpy.concatenate((Y, y), axis=0)

    if not supervised:
        constructUnsupervisedDataSet(X, Y, savePath, quantify)
    # else:
    #     constructSupervisedDataSet(X, Y, savePath, quantify, Labels)


def constructUnsupervisedDataSet(X, Y, savePath, quantify="",
                                 percentage=0.6):
    """
    only test have abnormal
    :param X:
    :param Y:
    :param quantify:
    :param savePath:
    :param percentage:
    :return:
    """
    LABLES = []
    AllN = X.shape[0]
    p_list = 0
    for i in range(80):
        count = numpy.sum(X[:, i] == 0)
        p_list += count
    test = p_list / X.shape[0]
    splitAbnormal = sum(Y == 0)
    # 归一化或标准化
    X_norm, labelName = chooseQuantify(X, quantify)
    X, X_AN = numpy.split(X_norm, [splitAbnormal], axis=0)

    # 合并Y
    Y = Y.reshape((Y.shape[0], 1))
    Y, Y_AN = numpy.split(Y, [splitAbnormal], axis=0)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=(1 - percentage) / 2, random_state=42)
    x_cross, x_test, y_cross, y_test = train_test_split(x_test, y_test, test_size=0.6, random_state=42)

    _, X_AN, _, Y_AN = train_test_split(X_AN, Y_AN, test_size=0.6,
                                        random_state=42)
    Y_AN = Y_AN.reshape(-1)
    print(Counter(Y_AN))
    Y_AN_TEMP = None
    X_AN_TEMP = None
    for i in range(1, 15):
        # 过滤
        # if i == 8 or i == 9:
        #     continue
        index = numpy.where((Y_AN == i))
        y_index = Y_AN[index]

        # 重修标签
        if 9 < i < 12:
            y_index = numpy.where(y_index == i, i - 2, 0)
        elif i == 12 or i == 13 or i == 14:
            y_index = numpy.where(y_index == i, 10, 0)
        elif i == 8:
            y_index = numpy.where(y_index == i, 11, 0)
        elif i == 9:
            y_index = numpy.where(y_index == i, 12, 0)
        else:
            y_index = numpy.where(y_index == i, i, 0)

        x_index = X_AN[index]
        print(y_index.shape, x_index.shape)
        # 减少过多样本的数量
        if i == 2 or i == 4 or i == 10:
            _, x_index, _, y_index = train_test_split(x_index, y_index, test_size=0.2,
                                                      random_state=42)
        if Y_AN_TEMP is None:
            Y_AN_TEMP = y_index
            X_AN_TEMP = x_index
        else:
            X_AN_TEMP = numpy.concatenate((X_AN_TEMP, x_index), axis=0)
            Y_AN_TEMP = numpy.concatenate((Y_AN_TEMP, y_index), axis=0)
    X_AN = deepcopy(X_AN_TEMP)
    Y_AN = deepcopy(Y_AN_TEMP)
    print(Counter(Y_AN))
    Y_AN = Y_AN.reshape((Y_AN.shape[0], 1))

    # get normal + abnormal test
    x_test_normal_shape = x_test.shape[0]
    x_test_abnormal_shape = X_AN.shape[0]
    x_test = numpy.concatenate((x_test, X_AN), axis=0)
    y_test = numpy.concatenate((y_test, Y_AN), axis=0)

    # 随机测试集
    x_test, y_test = random(x_test, y_test)

    # 修改y shape
    y_train = y_train.reshape(-1)
    y_cross = y_cross.reshape(-1)
    y_test = y_test.reshape(-1)

    labelName = labelName + "-unsupervised"

    print("%s num:%d, train num:%d, crossValidation num:%d, test num:%d" % (
        labelName, AllN, x_train.shape[0], x_cross.shape[0], x_test.shape[0]))
    LABLES.append(
        {"name": labelName,
         "num": AllN,
         "train_num": x_train.shape[0],
         "cross_num": x_cross.shape[0],
         "x_test_normal_shape": x_test_normal_shape,
         "x_test_abnormal_shape": x_test_abnormal_shape,
         "test_num": x_test.shape[0],
         }
    )

    # 保存成npz文件
    npz_path = os.path.join(savePath, labelName) + ".npz"
    if not os.path.exists(npz_path):
        numpy.savez(npz_path, x_train=x_train, y_train=y_train, x_cross=x_cross, y_cross=y_cross, x_test=x_test,
                    y_test=y_test)

    labels_path = savePath + "\\" + "labels.csv"
    p = pd.DataFrame(LABLES, dtype=int)
    p.to_csv(labels_path, mode='a+', index=None)


# def constructSupervisedDataSet(X, Y, savePath, quantify="", Labels=None,
#                                percentage=0.6):
#     """
#     train , cross ,test  always have abnormal
#     :param Labels:
#     :param X:
#     :param Y:
#     :param quantify:
#     :param savePath:
#     :param percentage:
#     :return:
#     """
#     LABLES = []
#     X_norm, labelName = chooseQuantify(X, quantify)
#     x_train, x_test, y_train, y_test = train_test_split(X_norm, Y, test_size=(1 - percentage) / 2, random_state=42)
#     x_train, x_cross, y_train, y_cross = train_test_split(x_train, y_train, test_size=(1 - percentage) / 2,
#                                                           random_state=42)
#
#     labelName = labelName + "-supervised"
#     print("%s num:%d, train num:%d, crossValidation num:%d, test num:%d" % (
#         labelName, X.shape[0], x_train.shape[0], x_cross.shape[0], x_test.shape[0]))
#     y_c = Counter(Y)
#     y_train_c = Counter(y_train)
#     y_cross_c = Counter(y_cross)
#     y_test_c = Counter(y_test)
#     for index, label in enumerate(Labels):
#         LABLES.append(
#             {"name": label,
#              "num": y_c.get(index),
#              "train_num": y_train_c.get(index),
#              "cross_num": y_cross_c.get(index),
#              "test_num": y_test_c.get(index),
#              }
#         )
#
#     # 保存成npz文件
#     npz_path = os.path.join(savePath, labelName) + ".npz"
#     if not os.path.exists(npz_path):
#         numpy.savez(npz_path, x_train=x_train, y_train=y_train, x_cross=x_cross, y_cross=y_cross, x_test=x_test,
#                     y_test=y_test)
#
#     labels_path = savePath + "\\" + labelName + "-labels.csv"
#     p = pd.DataFrame(LABLES, dtype=int)
#     p.to_csv(labels_path, mode='w', index=None)
#     pass


def chooseQuantify(X, quantify=""):
    # 归一化或标准化
    if quantify == "standard":
        labelName = "dataset-standard"
        X_norm = standardScaler(X)
    elif quantify == "L1":
        labelName = "dataset-normalize-L1"
        X_norm = normalize(X, "l1")
    elif quantify == "L2":
        labelName = "dataset-normalize-L2"
        X_norm = normalize(X, "l2")
    elif quantify == "minMax":
        labelName = "dataset-minMax"
        X_norm = minMaxScaler(X)
    elif quantify == "maxAbs":
        labelName = "dataset-maxAbs"
        X_norm = maxAbsScaler(X)
    else:
        labelName = "dataset-none"
        X_norm = X

    return X_norm, labelName


if __name__ == '__main__':
    classification("Classification")

    # constructDataSet(quantify="standard", srcPath="Classification", supervised=False)
    # constructDataSet(quantify="L1", srcPath="Classification", supervised=False)
    # constructDataSet(quantify="L2", srcPath="Classification", supervised=False)
    constructDataSet(quantify="minMax", srcPath="Classification", supervised=False)
    # constructDataSet(quantify="dataset-minMax", srcPath="Classification", supervised=False)
    # constructDataSet(quantify="maxAbs", srcPath="Classification", supervised=False)
