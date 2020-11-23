# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     IsolationForest
   Description :
   Author :       ybw
   date：          2020/11/22
-------------------------------------------------
   Change Activity:
                   2020/11/22:
-------------------------------------------------
"""

# -*- coding: utf-8 -*-
from collections import Counter

from scipy.stats import stats
from sklearn.ensemble import IsolationForest

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
import time

import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.metrics import classification_report, f1_score

from config.setting import dataset, modelPath
from utlis.file import mkdir
from utlis.sklearn import describe_loss_and_accuracy, describe_evaluation, distribution
import seaborn as sns
import matplotlib.pyplot as plt

from utlis.time import getTime


class isolationForest:
    """
    异常值是少量且不同的观测值，因此更易于识别。孤立森林集成了孤立树，在给定的数据点中隔离异常值
    """
    modelPath = os.path.join(modelPath, "IsolationForest")
    LABELS = None
    dataQuantification = None
    model = None
    time = getTime()

    def __init__(self, dataset, model=None):
        mkdir(self.modelPath)
        self.loadData(dataset)
        self.loadLabels(dataset)
        if model:
            self.loadModel(model)

    def loadLabels(self, name):
        path = os.path.join(dataset, "DeepLearningDateSet", name + "-labels.csv")
        df = pd.read_csv(path)
        name = df["name"]
        num = df["train_num"]
        self.LABELS = name.tolist()
        self.contaminationParameter = np.sum(num[1:]) / num[0]

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

    def getName(self):
        return "IsolationForest-" + self.dataQuantification + "-" + self.time

    def createModel(self):
        self.model = IsolationForest(n_estimators=400, max_samples=1024, contamination=self.contaminationParameter)

    def fit(self):
        if not self.model:
            self.createModel()
        # train 让模型更加适应正分类
        self.model.fit(self.x_train)

    def anomalyScore(self, model=None):
        scores = self.model.decision_function(self.x_test)
        distribution(scores, self.modelPath,self.getName())
        threshold = stats.scoreatpercentile(scores, 100 * self.contaminationParameter)

        print("-------------------------------------")
        print("threshold:", threshold)
        print(Counter(self.y_test))
        print(Counter(self.y_test[threshold > scores]))
        print("-------------------------------------")

        self.evaluate(scores, threshold)
        # print(Counter(y_val[cutoff > scores]))
        # scores_test = IF.decision_function(X_test)
        # print(Counter(y_test))
        # print(Counter(y_test[cutoff > scores_test]))

    def evaluate(self, score, Threshold):
        y_pred = np.where(score > Threshold, 0, 1)
        self.y_test = np.where(self.y_test > 0.5, 1, 0)
        describe_evaluation(self.y_test, y_pred, ["BENIGN", "MALIGNANT"],
                            self.modelPath, self.getName())

    def predict(self):
        outlier_label = self.model.fit_predict(self.x_test)
        outlier_label = np.where(outlier_label > 0, 0, 1)
        self.y_test = np.where(self.y_test > 0.5, 1, 0)
        describe_evaluation(self.y_test, outlier_label, ["BENIGN", "MALIGNANT"],
                            self.modelPath, self.getName())

    def save(self):
        pass
        # mkdir(self.modelPath)
        # self.model.save(os.path.join(self.modelPath, "sae_" + self.dataQuantification + "_" + getTime() + ".h5"))

    def loadModel(self, model):
        pass
        # model = os.path.join(self.modelPath, model)
        # if model and os.path.exists(model):
        #     self.model = load_model(model)


def main():
    sae = isolationForest("dataset-minMax-supervised")
    sae.fit()
    # sae.save()
    sae.predict()
    # sae.anomalyScore()


if __name__ == '__main__':
    main()
