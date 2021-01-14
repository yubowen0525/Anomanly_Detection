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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc

from utlis.file import mkdir
from utlis.time import getTime
import uuid


def minMaxScaler(x):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
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


import pylab as pl
def describe_loss_VAE(G_losses,D_losses, val_G_losses,val_D_losses, savePath=None, name=None):
    """
    :param loss:
    :param val_loss:
    :param savePath:
    :param name:
    :return:
    """
    plt.plot(numpy.arange(0, len(G_losses), 1).tolist(), G_losses, label='G_losses')
    plt.plot(numpy.arange(0, len(D_losses), 1).tolist(), D_losses, label='D_losses')
    plt.plot(numpy.arange(0, len(val_G_losses), 1).tolist(), val_G_losses, label='val_G_losses')
    plt.plot(numpy.arange(0, len(val_D_losses), 1).tolist(), val_D_losses, label='val_D_losses')
    # my_y_ticks = numpy.arange(6, 4, 0.5)
    # plt.yticks(my_y_ticks)
    plt.title("train epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    picturSave(savePath, name)
    plt.show()

    # x = numpy.arange(0, len(loss), 1)
    # p2 = pl.plot(x, loss, 'r-', label=u'loss')
    # pl.legend()
    # # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
    # p3 = pl.plot(x, val_loss, 'b-', label=u'val_loss')
    # plt.legend()
    # pl.xlabel(u'epoch')
    # pl.ylabel(u'loss')
    # plt.show()


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
        path = os.path.join(savePath, name, name + "-" + time + ".txt")
        classificationPath = os.path.join(savePath, name, name + "-" + time + ".txt")
        content = "{name}\t{time}\n{content}\n".format(name=name, time=time, content=content)
        saveFile(classificationPath, content)

    print(content)


def plot_roc(labels, predict_prob, savePath=None, name=None):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob, pos_label=1)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    picturSave(savePath, name)
    plt.show()


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
    sns.violinplot(x="class", y="score", data=df, split=True)
    picturSave(savePath, name)
    plt.show()


def boxplot(x, y, savePath, name):
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
    sns.boxplot(x="class", y="score", data=df)
    picturSave(savePath, name)
    plt.show()


def saveFile(path, content):
    mkdir(os.path.dirname(path))
    with open(path, "a+", encoding="utf-8") as f:
        f.write(content)
        f.write("\n")
