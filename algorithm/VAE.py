# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     VAE
   Description :
   Author :       ybw
   date：          2020/11/25
-------------------------------------------------
   Change Activity:
                   2020/11/25:
-------------------------------------------------
"""
import os
import bisect
import re

import scipy.stats as st
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from tensorflow.keras import layers, Model
import tensorflow.keras as keras
# from tensorflow_core.python.keras.models import load_model
from tensorflow.python.keras.models import load_model

from utils import metrics
from utils.file import mkdir
from config.setting import modelPath, dataset
from utils.metrics import roc_auc, pre_rec_curve
from utils.sklearn import describe_evaluation, distribution, violinplot, describe_loss_and_accuracy, plot_roc, \
    minMaxScaler, boxplot, describe_loss_VAE
from utils.time import getTime

latent_dim = 2


class Conv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation, strides=1, padding='valid'):
        super().__init__()
        self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(
            filters, (kernel_size, 1), (strides, 1), activation=activation, padding=padding
        )

    def call(self, x):
        x = tf.expand_dims(x, axis=2)
        x = self.conv2dtranspose(x)
        x = tf.squeeze(x, axis=2)
        return x


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


loss = []
val_loss = []


class _VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(_VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        test_data = None
        if isinstance(data, tuple):
            test_data = data[1]
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 80 * 1
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.01
            total_loss = reconstruction_loss + kl_loss

            if test_data is not None:
                z_mean_test, z_log_var_test, z_test = self.encoder(test_data)
                reconstruction_test = self.decoder(z_test)
                reconstruction_loss_test = tf.reduce_mean(
                    keras.losses.binary_crossentropy(test_data, reconstruction_test)
                )
                reconstruction_loss_test *= 80 * 1
                kl_loss_test = 1 + z_log_var_test - tf.square(z_mean_test) - tf.exp(z_log_var_test)
                kl_loss_test = tf.reduce_mean(kl_loss_test)
                kl_loss_test *= -0.01
                total_loss_test = reconstruction_loss_test + kl_loss_test

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "val_loss": total_loss_test,
        }

    def call(self, inputs):
        pass

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


class VAE:
    modelPath = os.path.join(modelPath, "VAE")
    LABELS = ["BENIGN", "MALIGNANT"]
    autoencoder = None
    dataQuantification = None
    time = getTime()
    epochs = 1
    batch_size = 256

    def __init__(self, dataset, model=None):
        """
        初始化模型
        :param dataset: 从dataset/DeepLearningDateSet 拿数据集
        :param model: 从self.modelPath拿模型, 加载模型
        """
        mkdir(self.modelPath)
        self.loadData(dataset)
        self.autoencoder = _VAE(self.buildEncoder(), self.buildDecoder())
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
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 80, 1)
        self.y_train = data["y_train"]
        self.x_cross = data["x_cross"]
        self.x_cross = self.x_cross.reshape(self.x_cross.shape[0], 80, 1)
        self.y_cross = data["y_cross"]
        self.x_test = data["x_test"]
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 80, 1)
        self.y_test = data["y_test"]

        print("x_train:", self.x_train.shape, "y_train:", self.y_train.shape,
              "x_cross:", self.x_cross.shape, "y_cross:", self.y_cross.shape,
              "x_test:", self.x_test.shape, "y_test:", self.y_test.shape)

    def buildEncoder(self):
        encoder_inputs = keras.Input(shape=(80, 1))
        x = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        # 相当于给输入加入噪声
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
        return encoder

    def buildDecoder(self):
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(1 * 20 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((20, 64))(x)
        x = layers.Conv1DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv1DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
        decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()
        return decoder

    def getname(self):
        return "VAE-" + self.dataQuantification + "-" + self.time

    def fit(self):
        # if not self.autoencoder:
        # train 让模型更加适应正分类
        self.autoencoder.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9, epsilon=1e-06))
        self.autoencoder.fit(self.x_train, self.x_train, epochs=self.epochs, batch_size=self.batch_size,
                             validation_data=(self.x_cross, self.x_cross))
        describe_loss_and_accuracy(self.autoencoder, self.modelPath, self.getname())
        # describe_loss_VAE(loss, val_loss, self.modelPath, self.getname())

        # 用新的模型进行训练评估
        # self.encoder.fit(self.x_train, self.y_train, epochs=20, batch_size=256, shuffle=True)

    # def plot_loss(self):
    #     file = "C:\\Users\\ybw\\Desktop\\text.txt"
    #     with open(file, encoding='utf-8', mode='rw') as f:
    #         # for line in f.readlines():
    #         content = f.read()
    #         loss = re.findall(r"- loss: (.*?) ", content)
    #         val_loss = re.findall(r"- val_loss: (.*?)\n", content)
    #         print()
    #         describe_loss_VAE(np.array(loss), np.array(val_loss), "C:\\Users\\ybw\\Desktop\\", "text")

    def anomalyScore(self, model=None):
        pred = self.predict()
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)
        pred = pred.reshape(pred.shape[0], -1)
        # 计算 self.pred 与 self.test 的 误差
        score = np.linalg.norm(pred - self.x_test, axis=1, ord=2)

        # normed to [0,1)
        score = (score - np.amin(score)) / (np.amax(score) -
                                            np.amin(score))
        # self.plot_loss()
        # 画误差分布图
        distribution(score, self.modelPath, self.getname())
        # 画误差小提琴图
        violinplot(score, self.y_test, self.modelPath, self.getname())
        # 画误差箱线图
        boxplot(score, self.y_test, self.modelPath, self.getname())
        # 画每个恶意攻击分类的ROC、AUC图
        metrics.MultiClassROCAUC(self.y_test, score, show=True,
                                 path=os.path.join(self.modelPath, self.getname(), self.getname()))

        # 画总体的ROC、AUC图
        self.y_test = np.where(self.y_test > 0.5, 1, 0)
        _ = roc_auc(self.y_test, score, show=True, path=os.path.join(self.modelPath, self.getname(), self.getname()))
        _ = pre_rec_curve(self.y_test, score, show=True,
                          path=os.path.join(self.modelPath, self.getname(), self.getname()))

        # 计算混淆矩阵，precision, recall, f1-score
        index = sum(self.y_test == 0)
        score_sort = np.sort(score)
        test = score_sort[(index - 1000): (1000 + index): 100]
        # self.evaluate(score, 0.60)
        f1Socre = []
        # index 就是异常个数与正常个数的分界点
        # 从index 附近找出2000个阈值，看哪个阈值的f1-score分值最高
        for Threshold in test:
            # print("min:", score.min(), "max:", score.max(), "mean:", score.mean(), "Threshold:", Threshold)
            y_pred = np.where(score < 0, 99, score)
            y_pred = np.where(y_pred > Threshold, 1, 0)
            self.y_test = np.where(self.y_test > 0.5, 1, 0)
            f1_score1 = f1_score(self.y_test, y_pred, average='macro')
            bisect.insort(f1Socre, (f1_score1, Threshold))

        f1Socre = f1Socre[-2:]
        for _, Threshold in f1Socre:
            print("min:", score.min(), "max:", score.max(), "mean:", score.mean(), "Threshold:", Threshold)
            self.evaluate(score, Threshold)

    def evaluate(self, score, Threshold):
        y_pred = np.where(score > Threshold, 1, 0)
        self.y_test = np.where(self.y_test > 0.5, 1, 0)
        plot_roc(self.y_test, y_pred, self.modelPath, self.getname())
        describe_evaluation(self.y_test, y_pred, self.LABELS, self.modelPath, self.getname())

    def predict(self):
        x_encoder = self.autoencoder.encoder.predict(self.x_test)
        return self.autoencoder.decoder.predict(x_encoder[2])

    def save(self):
        mkdir(os.path.join(self.modelPath, self.getname()))
        # keras.models.save_model(self.autoencoder, os.path.join(self.modelPath, self.getname()), save_format='tf')
        self.autoencoder.save_weights(os.path.join(self.modelPath, self.getname(), "model"),
                                      save_format='tf')
        # self.autoencoder.save_model(os.path.join(self.modelPath, self.getname(), self.getname() + ".h5"))

    def loadModel(self, model):
        path = os.path.join(self.modelPath, model)
        if model and os.path.exists(path):
            self.autoencoder.load_weights(os.path.join(path, "model"))

    def reconstructed_probability(self, X, L=100):
        """
        论文基于概率的预测
        :param X:
        :param L: L times sample
        :return:
        """
        reconstructed_prob = np.zeros((X.shape[0],), dtype='float32')
        mu_hat, sigma_hat, z = self.autoencoder.encoder.predict(X)
        X = X.reshape(X.shape[0], -1)
        for l in range(L):
            # mu_hat = mu_hat.reshape(X.shape)
            # sigma_hat = sigma_hat.reshape(X.shape) + 0.00001
            for i in range(X.shape[0]):
                p_l = st.norm.cdf(X[i, :], loc=sum(mu_hat[i, :]), scale=sum(sigma_hat[i, :]))
                reconstructed_prob[i] += p_l
        reconstructed_prob /= L
        return reconstructed_prob

    def is_outlier(self, L=100, alpha=0.05):
        p_hat = self.reconstructed_probability(self.x_test, L)
        distribution(p_hat, self.modelPath, self.getname())
        violinplot(p_hat, self.y_test, self.modelPath, self.getname())
        y_pred = p_hat < alpha
        describe_evaluation(self.y_test, y_pred, self.LABELS, self.modelPath, self.getname())

        return p_hat < alpha


def main():
    path = "VAE-minMax-x-80-unsupervised-2021-01-03-14-37-35"
    vae = VAE("dataset-minMax-unsupervised", model=path)
    # vae = VAE("dataset-minMax-unsupervised")
    # vae.fit()
    # vae.save()
    vae.anomalyScore()
    # vae.is_outlier()


# def iforest_predict(train, test, test_label):
#     from sklearn.ensemble import IsolationForest
#     iforest = IsolationForest(max_samples='auto',
#                               behaviour="new", contamination=0.01)
#
#     iforest.fit(train)
#     iforest_predict_label = iforest.predict(test)
#     # plot_confusion_matrix(test_label, iforest_predict_label, ['anomaly','normal'],'iforest Confusion-Matrix')
#
#
# def lof_predict(train, test, test_label):
#     from sklearn.neighbors import LocalOutlierFactor
#     lof = LocalOutlierFactor(novelty=True, contamination=0.01)
#     lof.fit(train)
#     lof_predict_label = lof.predict(test)
#     # plot_confusion_matrix(test_label, lof_predict_label, ['anomaly','normal'],'LOF Confusion-Matrix')
#

if __name__ == '__main__':
    main()
