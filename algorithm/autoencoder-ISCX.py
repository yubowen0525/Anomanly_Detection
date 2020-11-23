# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:06:43 2018

@author: 13913


"""

'''
Created on 19/12/2017
@author: rjpg
'''


from keras.datasets import mnist 
from keras.models import Model 
from keras.layers import Input, Dense 
from keras.utils import np_utils 
import numpy as np
from tensorflow.python.ops.variables import trainable_variables
import matplotlib.pyplot as plt
import pandas as pd


num_classes = 15

## Importing the dataset
#dataset = pd.read_csv('D:\\pcapcsv\\all.csv')
#dataset = pd.read_csv('C:\\csv2\\merge-data\\all_matlab.csv')

dataset = pd.read_csv('./result/smote_balance_15.csv')  #
#dataset = pd.read_csv('C:\\csv2\\merge-data\\balanced\\All_balance_15.csv')  #


print (dataset.shape)

# Importing the dataset
#labelset = pd.read_csv('C:\\csv2\\label\\Label_15_app.csv')

labelset = pd.read_csv('./result/Label_smote_balance_15.csv')      #
#labelset = pd.read_csv('C:\\csv2\\label\\balanced\\Label_15_balanced.csv')      #

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
print(Y_train.shape)


print('******** Y1_Test ')
print(Y_test.shape)



Y1_train = Y_train
print('******** Y1_TRAIN ')
print(Y1_train.shape)

Y1_test = Y_test
print('******** Y1_Test ')
print(Y1_test.shape)




#Convert data type and normalize valuesPython
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


input_img = Input(shape=(1480,))

x = Dense(1480, activation='relu')(input_img)

encoded1 = Dense(740, activation='relu')(x)
encoded2 = Dense(92, activation='relu')(encoded1)


y = Dense(32, activation='relu')(encoded2)


decoded2 = Dense(92, activation='relu')(y)
decoded1 = Dense(740, activation='relu')(decoded2)

z = Dense(1480, activation='sigmoid')(decoded1)
autoencoder = Model(input_img, z)



#encoder is the model of the autoencoder slice in the middle 
encoder = Model(input_img, y)

autoencoder.compile(optimizer='adadelta', loss='mse') # reporting the loss


from keras.callbacks import ModelCheckpoint, TensorBoard

checkpointer = ModelCheckpoint(filepath="./MLPmodel/modelSAE_model_15APP_gpu_ib.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./MLPmodel/SAE_imbalanced',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)



autoencoder.fit(X_train, X_train, epochs=50, batch_size=128)

# history = autoencoder.fit(X_train, X_train,
#       epochs=50,
#       batch_size=256,
#       shuffle=True,
#       validation_data=(X_test, X_test),
#       callbacks=[checkpointer, tensorboard]).history

# if you want an encoded flatten representation of every test MNIST
reduced_representation =encoder.predict(X_test)

#print encoded1 weights
#weights = autoencoder.layers[1].get_weights() # list of numpy arrays
#print(weights)

# if you want to lock the weights of the encoder on post-training 
#for layer in encoder.layers : layer.trainable = False


# define new model encoder->Dense  10 neurons with soft max for classification 
out2 = Dense(num_classes, activation='softmax')(encoder.output)
newmodel = Model(encoder.input,out2)


newmodel.compile(loss='categorical_crossentropy',
          optimizer='adam', 
          metrics=['accuracy']) 


checkpointer2 = ModelCheckpoint(filepath="./MLPmodel/modelSAE_model2_15APP_gpu_ib.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard2 = TensorBoard(log_dir='./MLPmodel/SAE_imbalanced2',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
newmodel.fit(X_train, Y1_train, epochs=100, batch_size=64)
newmodel.save('./MLPmodel/SMOTE-SAEmodel.h5')
# history2 = newmodel.fit(X_train, Y1_train,
#       epochs=100,
#       batch_size=256,
#       shuffle=True,
#       validation_data=(X_test, Y1_test),
#       callbacks=[checkpointer2, tensorboard2]).history

#print encoded1 weights again 
#weights = newmodel.layers[1].get_weights() # list of numpy arrays
#print(weights)


scores = newmodel.evaluate(X_test, Y1_test, verbose=1) 
print("Accuracy: ", scores[1])


import seaborn as sns
from pylab import rcParams

pred = newmodel.predict(X_test)

y_classes = pred.argmax(axis = -1)

y_true = Y1_test.argmax(axis = -1)

# from sklearn.metrics import confusion_matrix
# conf_matrix = confusion_matrix(y_true,y_classes)
# print(conf_matrix)
#
# sns.set(style='whitegrid', palette='muted', font_scale=2.4)
# sns.set_style("white")
# rcParams['figure.figsize'] = 14, 8
# RANDOM_SEED = 42
# LABELS = ["aim_chat", "email","facebook", "gmail","hangout", "ICQ","netflix", "scpDown","sftpDown", "skype","spotify", "torTwitter","vimeo", "voipbuster","youtube"]
#
# plt.figure(figsize=(24, 24))
# sns.heatmap(conf_matrix, cmap="Oranges", xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", linewidths=0.2, linecolor = 'black', cbar=False);
# plt.title("Traffic Classification Confusion matrix (SAE method)")
# plt.ylabel('application traffic samples')
# plt.xlabel('application traffic samples')
# plt.show()
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_true,y_classes)
print(conf_matrix)

sns.set(style='whitegrid', palette='muted', font_scale=2.4)
sns.set_style("white")
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["aim_chat", "email","facebook", "gmail","hangout", "ICQ","netflix", "scpDown","sftpDown", "skype","spotify", "torTwitter","vimeo", "voipbuster","youtube"]

plt.figure(figsize=(24, 24))
sns.heatmap(conf_matrix, cmap="Oranges", xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", linewidths=0.2, linecolor = 'black', cbar=False);
plt.title("Traffic Classification Confusion matrix (SAE method)")
plt.ylabel('application traffic samples')
plt.xlabel('application traffic samples')
plt.show()








