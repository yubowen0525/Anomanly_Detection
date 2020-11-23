# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     sklearn
   Description :
   Author :       ybw
   date：          2020/11/17
-------------------------------------------------
   Change Activity:
                   2020/11/17:
-------------------------------------------------
"""
import os

import numpy
import pandas
from matplotlib import rcParams
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from utlis.file import mkdir
from utlis.time import getTime
import uuid


def minMaxScaler(x):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_Normalized = min_max_scaler.fit_transform(x)
    return X_Normalized


def maxAbsScaler(x):
    max_abs_scaler = preprocessing.MaxAbsScaler()
    X_Normalized = max_abs_scaler.fit_transform(x)
    return X_Normalized


def normalize(x, norm="l2"):
    return preprocessing.normalize(x, norm=norm)


def picturSave(savePath, name):
    if savePath:
        picturePath = os.path.join(savePath, name, name + uuid.uuid1().hex + ".png")
        mkdir(os.path.join(savePath, name))
        plt.savefig(picturePath)


def standardScaler(x):
    """
    使数据符合正太分布
    :param x:
    :return:
    """
    min_max_scaler = preprocessing.StandardScaler()
    X_Standarded = min_max_scaler.fit_transform(x)
    return X_Standarded


def describe_loss_and_accuracy(model, savePath=None, name=None):
    """
    描述训练集与测试集的acc,loss看是否欠拟合，过拟合
    :param savePath:
    :param model:
    :return:
    """
    if hasattr(model, "history"):
        # loss
        history = model.history
        if history.history.get('loss'):
            plt.plot(history.epoch, history.history.get('loss'), label='loss')
            plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
            plt.legend()

        # accuracy
        if history.history.get('accuracy'):
            plt.plot(history.epoch, history.history.get('accuracy'), label='accuracy')
            plt.plot(history.epoch, history.history.get('val_accuracy'), label='val_accuracy')
            plt.legend()
        picturSave(savePath, name)
        plt.show()
    else:
        print("No training history！")


def describe_evaluation(y_true, y_pred, LABELS, savePath=None, name=None):
    """
    confusion_matrix and f1-score
    :param y_true: true labels
    :param y_pred: pred labels
    :param LABELS: type is list.  example [""]
    :param savePath:
    :return:
    """
    # 混合矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    # f, ax = plt.subplots()
    rcParams['figure.figsize'] = 10, 6
    # RANDOM_SEED = 42
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", cmap='GnBu')
    plt.title("Traffic Classification Confusion Matrix (SAE method)")
    plt.ylabel('Application traffic samples')
    plt.xlabel('Application traffic samples')
    time = getTime()
    picturSave(savePath, name)
    plt.show()

    # precision, recall, f1-score, support
    content = classification_report(y_true, y_pred, target_names=LABELS, digits=4)
    if savePath:
        classificationPath = os.path.join(savePath, name, name + "-" + time + ".txt")
        with open(classificationPath, "a+", encoding="utf-8") as f:
            f.write(name + "\t" + time + "\n")
            f.write(content)
            f.write("\n")

    print(content)


def random(x, y):
    X_Y = numpy.concatenate((x, y), axis=1)
    numpy.random.shuffle(X_Y)

    # 切分
    return numpy.split(X_Y, [X_Y.shape[1] - 1], axis=1)


def distribution(y, savePath, name):
    # RANDOM_SEED = 42
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    sns.kdeplot(y, shade=True, ax=ax)
    picturSave(savePath, name)
    plt.show()


def scatterplot(x, y):
    x = x.reshape((x.shape[0], 1))
    y = y.reshape((y.shape[0], 1))
    data = numpy.concatenate([x, y], axis=1)
    df = pandas.DataFrame(data, columns=["x", "y"])
    sns.scatterplot(x="x", y="y", data=df)
    # sns.jointplot(x="x", y="y", data=df)
    plt.show()


def violinplot(x, y, savePath, name):
    """
    这里x是横坐标，种类，y是纵坐标
    :param x:
    :param y:
    :return:
    """
    plt.figure(figsize=(24, 20))
    x = x.reshape((x.shape[0], 1))
    y = y.reshape((y.shape[0], 1))
    data = numpy.concatenate([y, x], axis=1)
    df = pandas.DataFrame(data, columns=["class", "score"])
    sns.violinplot(x="class", y="score", data=df)
    picturSave(savePath, name)
    plt.show()
