import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense,MaxPooling1D,AveragePooling1D,Conv1D
from keras.models import Model
from keras.layers.core import Dropout, Activation, Flatten
import time
import gc
import sklearn.metrics
from sklearn.metrics import classification_report,f1_score
from sklearn.metrics import accuracy_score
from keras.layers import Dense, Input, dot, merge
from keras.models import Model, Sequential
from keras import regularizers
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, LSTM, Conv3D, MaxPool3D, Conv1D, MaxPool1D, SpatialDropout3D, concatenate
from keras.layers import SeparableConv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
from keras.utils.np_utils import to_categorical
from keras import optimizers
import csv
from sklearn.metrics import confusion_matrix
import scipy.io as sio
import os
import random

from get_label import get_label
from read_other import get_roi_signal
from read_network import *


def get_conn(datadir):
    y = get_label()
    Y = np.reshape(y, [-1, 1])
    subjects_path = os.listdir(datadir)
    subjects_path.sort()
    datapath = []
    feature = []
    for file in subjects_path:
        datapath.append(os.path.join(datadir, file))
    for subject in datapath:
        mat = sio.loadmat(subject)['dynamic_corr']
        idx = np.triu_indices_from(mat[0], 1)
        vec_networks = [m[idx] for m in mat]
        vec_networks = np.array(vec_networks)
        vec_networks = vec_networks.astype('float64')
        feature.append(vec_networks)
    data_x = np.array(feature)
    data_y = np.array(Y)
    return data_x,data_y

def CNNLSTM_model(step, fea):
    main_input = Input(shape=(step,fea,), name='main_input')
    x = Conv1D(filters=500,kernel_size=3,name='conv1',kernel_initializer=initializers.glorot_normal())(main_input)
    x = BatchNormalization(name='bn1')(x)
    x = LeakyReLU()(x)
    x = Conv1D(filters=128,kernel_size=3,name='conv2',kernel_initializer=initializers.glorot_normal())(x)
    x = BatchNormalization(name='bn2')(x)
    x = LeakyReLU()(x)
    x = MaxPooling1D()(x)
    x = LSTM(64,activation='tanh', name='lstm1',kernel_initializer=initializers.glorot_normal(),dropout=0.5, recurrent_dropout=0.5,return_sequences=True)(x)
    x = LSTM(32,activation='tanh',name='lstm2',kernel_initializer=initializers.glorot_normal(),dropout=0.25, recurrent_dropout=0.25,return_sequences=False)(x)
    x = Dense(128, name='dense',kernel_initializer=initializers.glorot_normal(),activation="tanh")(x)
    preds = Dense(1, activation='sigmoid',name='pred')(x)
    model = Model(main_input, preds)
    return model


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn import preprocessing
from scipy import interp
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import keras.backend.tensorflow_backend as KTF

import warnings

warnings.filterwarnings('ignore')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
tprs = []
aucs = []
all_result = []
mean_fpr = np.linspace(0, 1, 100)
k = 0
ntrue = []
all_true_sample = []
all_true_label = []
#subject_list = np.array(sub_list)
X , Y = get_conn("/data/hym/MDD/dynamic_mat")
del_list = []
for i in range(X.shape[0]):
    #print(X[i].shape[0])
    if X[i].shape[0] < 150:
        del_list.append(i)
X = np.delete(X,del_list,axis=0)
Y = np.delete(Y,del_list,axis=0)
feat1 = []
for i in range(X.shape[0]):
    feat1.append(X[i][:151])
feat1 = np.array(feat1)
X = feat1


for train_index, test_index in skf.split(X, Y):
    print("###################fold ", k, "#################")
    x_train = X[train_index]
    y_train = Y[train_index]

    x_test = X[test_index]
    y_test = Y[test_index]
    print('train shape:', x_train.shape, 'test shape:', x_test.shape)
    # x_train, x_test = X[train_index], X[test_index]
    # y_train, y_test = label_list[train_index], label_list[test_index]
    print("train MDD, HC sample:", y_train.tolist().count(1), y_train.tolist().count(0))
    print("test MDD, HC sample:", y_test.tolist().count(1), y_test.tolist().count(0))

    dim1 = x_train.shape[1]
    dim2 = x_train.shape[2]
    model = CNNLSTM_model(dim1, dim2)
    my_adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=my_adam, loss='binary_crossentropy', metrics=['accuracy'])
    # checkpoint
    bestModelSavePath = 'lstm/fold' + str(k) + '_lstm_aal.hdf5'
    checkpoint = ModelCheckpoint(bestModelSavePath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

    startTime = time.time()

    model.fit(x_train, y_train, epochs=100, batch_size=10, validation_data=(x_test, y_test), callbacks=[checkpoint])

    endTime = time.time()
    durTime = endTime - startTime
    print('&&& Time: ', durTime)

    # test
    predNoneFlag = True
    model.load_weights(bestModelSavePath)
    print('Start Predict.')
    pred = model.predict(x_test)
    print("pred shape:", pred.shape, "y_true shape", y_test.shape)
    del x_train, x_test
    gc.collect()
    y_pred = []
    for p in pred:
        y_pred.append(round(p[0]))
    y_pred = np.array(y_pred)

    print("y_pred shape:", y_pred.shape)
    print("y_pred:", y_pred[:5])

    true_list = []
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            true_list.append(i)
    true_list = np.array(true_list)
    ntrue.append(len(true_list))
    print('true_list shape:', true_list.shape)

    all_true_label.append(y_test[true_list])

    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuary: %.2f%%" % (test_accuracy * 100.0))
    target_names = ['class 0', 'class 1']
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    SS = tp / (tp + fn)
    SC = tn / (tn + fp)
    GR = (tp + tn) / (tp + fn + tn + fp)
    f1 = f1_score(y_test, y_pred)
    print("SS:", SS)
    print("SC:", SC)
    print("GR:", GR)
    print("f1_macro:", f1)
    print(model.summary())

    print('dnn_method time : ', durTime)

    result = [SS, SC, GR, f1]
    all_result.append(result)

    # compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROc fold %d(AUC = %0.4f)' % (k, roc_auc))
    k += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label='Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig("result.jpg")
plt.show()
print(all_result)
print(np.mean(all_result, axis=0).tolist())
print(np.std(all_result, axis=0).tolist())