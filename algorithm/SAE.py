# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     SAE
   Description :
   Author :       ybw
   date：          2020/11/18
-------------------------------------------------
   Change Activity:
                   2020/11/18:
-------------------------------------------------
"""
import bisect
import os

# from tensorflow.keras import models
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Dropout
import time

from keras import regularizers
from keras.models import Model, load_model
from keras.layers import Dense, Input
import numpy as np
from matplotlib import rcParams
from sklearn.metrics import classification_report, precision_score

from config.setting import dataset, modelPath
from utlis.file import mkdir
from utlis.sklearn import describe_loss_and_accuracy, describe_evaluation, distribution, scatterplot, violinplot
import seaborn as sns
import matplotlib.pyplot as plt

from utlis.time import getTime


class SAE:
    modelPath = os.path.join(modelPath, "SAE")
    LABELS = ["BENIGN", "MALIGNANT"]
    autoencoder = None
    dataQuantification = None
    time = getTime()

    def __init__(self, dataset, model=None):
        mkdir(modelPath)
        self.loadData(dataset)
        if model:
            self.loadModel(model)

    def loadData(self, name="dataset"):
        """
        3个数据集：训练集60%，交叉验证集合20%，测试集20%
        :param name:
        :return:
        """
        try:
            self.dataQuantification = "-".join(name.split("-")[1:])
        except IndexError:
            print("数据集需要数据量化方式")
            exit(1)
        path = os.path.join(dataset, "DeepLearningDateSet", name)
        data = np.load(path + ".npz")
        self.x_train = data["x_train"]
        self.y_train = data["y_train"]
        self.x_cross = data["x_cross"]
        self.y_cross = data["y_cross"]
        self.x_test = data["x_test"]
        self.y_test = data["y_test"]

        print("x_train:", self.x_train.shape, "y_train:", self.y_train.shape,
              "x_cross:", self.x_cross.shape, "y_cross:", self.y_cross.shape,
              "x_test:", self.x_test.shape, "y_test:", self.y_test.shape)

    def getname(self):
        return "SAE-" + self.dataQuantification + "-" + self.time

    def createModel(self):
        input_data = Input(shape=(78,))

        # 编码层
        encoded = Dense(32, activation='relu', activity_regularizer=regularizers.l1(10e-5), name='encoded_hidden1')(
            input_data)
        # encoded = Dense(256, activation='relu', activity_regularizer=regularizers.l1(10e-5), name='encoded_hidden2')(
        #     encoded)
        # encoded = Dense(8, activation='relu', activity_regularizer=regularizers.l1(10e-5),
        #                 name='encoded_hidden2')(encoded)
        y = Dense(2, activation='relu', activity_regularizer=regularizers.l1(10e-5),
                  name='encoded_hidden3')(encoded)
        # LR = Dense(2, activation='softmax', name='LR')(encoder_output)

        # 解码层
        # decoded = Dense(8, activation='relu', name='decoded_hidden1')(y)
        # decoded = Dense(16, activation='relu', name='decoded_hidden2')(decoded)
        decoded = Dense(32, activation='relu', name='decoded_hidden1')(y)
        decoded = Dense(78, activation='tanh', name='decoded_output2')(decoded)

        # 构建自编码模型
        self.autoencoder = Model(inputs=input_data, outputs=decoded)

        # complile autoencoder 设置自编码的优化参数
        self.autoencoder.compile(optimizer='adam', loss='mse')
        # self.autoencoder.compile(optimizer='sgd', loss='mse')

        print(self.autoencoder.summary())

        # self.encoder = Model(inputs=input_data, outputs=LR)
        # self.encoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        # print(self.encoder.summary())

    def fit(self):
        if not self.autoencoder:
            self.createModel()
        # train 让模型更加适应正分类
        self.autoencoder.fit(self.x_train, self.x_train, epochs=70, batch_size=4096, shuffle=True,
                             validation_data=(self.x_cross, self.x_cross))
        describe_loss_and_accuracy(self.autoencoder, self.modelPath, self.getname())

        # 用新的模型进行训练评估
        # self.encoder.fit(self.x_train, self.y_train, epochs=20, batch_size=256, shuffle=True)

    def anomalyScore(self, model=None):
        pred = self.predict()
        # 计算 pred 与 测试集的 欧几里得距离
        score = np.linalg.norm(pred - self.x_test, axis=1, ord=np.inf)
        distribution(score, self.modelPath, self.getname())
        violinplot(score, self.y_test, self.modelPath, self.getname())
        index = sum(self.y_test == 0)
        score_sort = np.sort(score)
        # 取出数据分布分界线中心1万的200个测试
        test = score_sort[(index - 1000): (1000 + index): 100]
        # self.evaluate(score, 0.60)
        # f1Socre = []
        # for Threshold in test:
        #     # print("min:", score.min(), "max:", score.max(), "mean:", score.mean(), "Threshold:", Threshold)
        #     y_pred = np.where(score > Threshold, 1, 0)
        #     self.y_test = np.where(self.y_test > 0.5, 1, 0)
        #     f1_score1 = f1_score(self.y_test, y_pred, average='macro')
        #     bisect.insort(f1Socre, (f1_score1, Threshold))

        # f1Socre = f1Socre[-2:]
        for Threshold in test:
            print("min:", score.min(), "max:", score.max(), "mean:", score.mean(), "Threshold:", Threshold)
            self.evaluate(score, Threshold)

    def evaluate(self, score, Threshold):
        y_pred = np.where(score > Threshold, 1, 0)
        self.y_test = np.where(self.y_test > 0.5, 1, 0)
        describe_evaluation(self.y_test, y_pred, self.LABELS, self.modelPath, self.getname())

    def predict(self):
        return self.autoencoder.predict(self.x_test)

    def save(self):
        mkdir(os.path.join(self.modelPath, self.getname()))
        self.autoencoder.save(os.path.join(self.modelPath, self.getname(), self.getname() + ".h5"))

    def loadModel(self, model):
        model = os.path.join(self.modelPath, model)
        if model and os.path.exists(model):
            self.autoencoder = load_model(model)


def main():
    sae = SAE("dataset-minMax", "SAE-minMax-2020-11-23-19-02-47\\SAE-minMax-2020-11-23-19-02-47.h5")
    # sae.fit()
    # sae.save()
    sae.anomalyScore()


if __name__ == '__main__':
    main()
