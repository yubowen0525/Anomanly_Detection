# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 03:45:47 2018

@author: blueboy
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
from pylab import rcParams


## Importing the dataset
#dataset = pd.read_csv('D:\\pcapcsv\\all.csv')
#dataset = pd.read_csv('C:\\csv2\\merge-data\\imbalanced\\All_unbalance_15.csv')  #不平衡数据集 20万条

dataset = pd.read_csv('.\\dataset\\01_20190314_4_784_all.csv')  #平衡数据集 7万条


print(dataset.shape)

# Importing the dataset
#labelset = pd.read_csv('C:\\csv2\\label\\imbalanced\\Label_15_imbalanced.csv')      #不平衡数据集label

labelset = pd.read_csv('.\\dataset\\02_20190314_4_784_all.csv')      #不平衡数据集label



print (labelset.shape)


#X = dataset.iloc[:, 0:1479].values
#y = labelset.iloc[:, 0].values
X = dataset
y = labelset



#y = np_utils.to_categorical(y, 3)

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

y_train_transpose = np.transpose(y_train)
y_train_transpose = y_train_transpose.values
y_test_transpose = np.transpose(y_test)
y_test_transpose = y_test_transpose.values

Y_train = np_utils.to_categorical(y_train_transpose[0])
Y_test = np_utils.to_categorical(y_test_transpose[0])

print('******** Y_TRAIN ')
#Y_train = Y_train[:,1:5]
print(Y_train.shape)
# print(Y_train)

print('******** Y1_Test ')
print(Y_test.shape)
# print(Y_test)


Y1_train = Y_train[:,1:6]
print('******** Y1_TRAIN ')
print(Y1_train.shape)
# print(Y1_train)


Y1_test = Y_test
print('******** Y1_Test ')
print(Y1_test.shape)
# print(Y1_test)



#Convert data type and normalize valuesPython
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


#Initializing Neural Network
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu', input_dim = 784))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'softmax'))

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

'''
增加TensorBoard相关记录
'''
from keras.callbacks import ModelCheckpoint, TensorBoard

checkpointer = ModelCheckpoint(filepath=".\\model\\MLP_balanced\\MLP_model_4APP_balanced_cpu.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='.\\tblog\\MLP\\balanced',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)


classifier.fit(X_train, Y_train, epochs=10, batch_size=256)
classifier.save('MLPmodel/MLP_model_4APP_real_gpu.h5')

# history = classifier.fit(X_train, Y1_train,
#       epochs=50,
#       batch_size=10,
#       shuffle=True,
#       validation_data=(X_test, Y1_test),
#       callbacks=[checkpointer, tensorboard]).history


# Fitting our model
#classifier.fit(X_train, Y1_train, batch_size = 20, nb_epoch = 10,callbacks=[TensorBoard(log_dir='C:\\csv2\\tblog\\FC')])

score = classifier.evaluate(X_test, Y1_test, batch_size=128)


#from keras.utils import plot_model
#plot_model(classifier, to_file='model.png')

# Output the confusion matrix
pred = classifier.predict_classes(X_test,batch_size=50)
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test_transpose[0],pred)
print(conf_matrix)

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 4, 8
RANDOM_SEED = 42
LABELS = ["aim_chat","hangout", "ICQ","netflix"]

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()









