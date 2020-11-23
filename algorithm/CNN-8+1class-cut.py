import sys
sys.path.append("../../")

import numpy as np

import tensorflow as tf

from tensorflow.python.keras import callbacks
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
import tfkerassurgeon
from tfkerassurgeon import identify
from tfkerassurgeon.operations import delete_channels
import pandas as pd
from tensorflow.python.keras.models import load_model



#设置系统参数
print(tf.__version__)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)

#超参数
#设置一些静态值，可以进行调整以进行实验
keras_verbosity = 2 # limits the printed output but still gets the Epoch stats 限制打印输出，但仍获取每轮的统计
epochs=200 # we'd never reach 200 because we have early stopping 我们不可能到达200，因为我们要提前停止
batch_size=128 # tweak this depending on your hardware and Model 根据您的硬件和型号进行调整
num_classes=9


#step 1 加载数据集
dataset = pd.read_csv('./dataset/01_20201108_8+1_int.csv')  #不平衡数据集 20万条
print (dataset.shape)
labelset = pd.read_csv('./dataset/02_20201108_8+1_int.csv')      #不平衡数据集label
print (labelset.shape)

X = dataset
y = labelset

#step 2 拆分数据集
from keras.utils import np_utils

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X, y, test_size = 0.5)


#step3  数据预处理
y_train_transpose = np.transpose(Y_train)
y_train_transpose = y_train_transpose.values
y_test_transpose = np.transpose(Y_test)
y_test_transpose = y_test_transpose.values

Y_train = np_utils.to_categorical(y_train_transpose[0])
Y_test = np_utils.to_categorical(y_test_transpose[0])

print('******** Y_TRAIN ')
print(Y_train.shape)

print('******** Y_Test ')
print(Y_test.shape)

print(X_train)

#Convert data type and normalize valuesPython
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

#评估模型
def eval_model(model):

    return model.evaluate(
        X_test,
        Y_test,
        batch_size=batch_size,
        verbose=keras_verbosity)

#按名称获取layer
def prune_layer_by_name(model, layer_name):

    # First we get the layer we are working on
    layer = model.get_layer(name=layer_name)
    # Then prune is and return the pruned model
    return prune_layer(model, layer)


# 此方法使用Keras surgeon 确定可以修剪图层的哪些部分，然后将其删除
# 注意：它将返回经过重新编译的新模型
def prune_layer(model, layer):

    # Get the APOZ (Average Percentage of Zeros) that should identify where we can prune
    # 获取APOZ（零的平均百分比），该数量应标识我们可以修剪的位置
    apoz = identify.get_apoz(model, layer, X_test)

    # Get the Channel Ids that have a high APOZ, which indicates they can be pruned
    # 获取具有较高APOZ的通道ID，表明它们可以被修剪
    high_apoz_channels = identify.high_apoz(apoz)

    # Run the pruning on the Model and get the Pruned (uncompiled) model as a result
    # 在模型上运行修剪并得到修剪（未编译）模型
    model = delete_channels(model, layer, high_apoz_channels)

    # Recompile the model
    # 重新编译模型
    compile_model(model)

    return model

# Simple reusable shorthand to compile the model, so that we can be sure to use the same optomizer, loss, and metrics
# 简单的可重用速记来编译模型，因此我们可以确保使用相同的优化器，损失和度量标准
def compile_model(model):

    rmsprop=optimizers.rmsprop(lr=0.0001, rho=0.9, epsilon=1e-06)
    model.compile(loss='binary_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])


# 一种简单的方法，用于获取回调进行训练
def get_callbacks(use_early_stopping = True, use_reduce_lr = True):

    callback_list = []

    if(use_early_stopping):

        callback_list.append(callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=0,
                                                     patience=10,
                                                     verbose=keras_verbosity,
                                                     mode='auto'))

    if(use_reduce_lr):

        callback_list.append(callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.1,
                                                         patience=5,
                                                         verbose=keras_verbosity,
                                                         mode='auto',
                                                         epsilon=0.0001,
                                                         cooldown=0,
                                                         min_lr=0))

    return callback_list

# 获取回调
callback_list = get_callbacks()

# Simple reusable shorthand for evaluating the model on the Validation set
# 简单可重复使用的速记，用于在验证集上评估模型
def fit_model(model):

    return model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=keras_verbosity,
        validation_data=(X_test, Y_test),
        callbacks=callback_list)


# 封装模型的体系结构和构造的方法
def build_model():

    # Create  model
    model=models.Sequential()
    model.add(layers.Convolution1D(64,3,padding="same",activation="relu",input_shape=(78,1),name='conv_1'))
    model.add(layers.Convolution1D(64,3,padding="same",activation="relu",name='conv_2'))
    model.add(layers.MaxPooling1D(pool_size=(2),name='maxpool_1'))
    model.add(layers.Convolution1D(128,3,padding="same",activation="relu",name='conv_3'))
    model.add(layers.Convolution1D(128,3,padding="same",activation="relu",name='conv_4'))
    model.add(layers.MaxPool1D(pool_size=(2),name='maxpool_2'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128,activation="relu",name='dense_1'))
    model.add(layers.Dropout(0,5))
    model.add(layers.Dense(num_classes,activation="sigmoid",name='dense_2'))


    rmsprop=optimizers.rmsprop(lr=0.0001, rho=0.9, epsilon=1e-06)
    model.compile(loss='binary_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])

    return model



def main():


    #load the model 若是加载模型，这步放开
    model = load_model('./model/CNNmodel8+1class_sigmoid_1108_cutted9.h5')

    # build the model ,若是加载模型不需要这步
    #model = build_model()

    # Initial Train on dataset
    # 数据集的初始训练 ,若是加载模型不需要这步
    #results = fit_model(model)

    # eval and print the results of the training
    loss = eval_model(model)
    print('original model loss:', loss, '\n')
    #model.save('./model/CNNmodel8+1class_sigmoid_1014.h5')
    #需要剪枝的layer
    layers_list=['conv_1','conv_2','conv_3','conv_4','dense_1']
    model1=model
    for i in layers_list:
        print("i",i)
        layer_name=i
        model1 = prune_layer_by_name(model1, layer_name)

    # eval and print the results of the pruning
    loss = eval_model(model1)
    print('model loss after pruning: ', loss, '\n')

    # Retrain the model to accomodate for the changes
    results = fit_model(model1)

    # eval and print the results of the retraining
    loss = eval_model(model1)
    model1.save('./model/CNNmodel8+1class_sigmoid_1108_cutted10.h5')
    print('model loss after retraining: ', loss, '\n')



# Run the main Method
if __name__ == '__main__':
    main()