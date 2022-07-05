import os
import re

import keras
import scipy.io as sio
import numpy as np
import pandas as pd
import nilearn
import sklearn.metrics
from nilearn import masking, plotting, image, datasets, input_data
from nilearn.image import resample_to_img
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
import scipy.io as scio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import pandas as pd

from get_label import get_label
from read_other import get_roi_signal
from read_network import get_feature
from model import *
from select_sub import select_sub

start = time.perf_counter()

filelist = ['ROISignals_ROISignal_ALFF_FunImgARglobalCW.mat',
            'ROISignals_ROISignal_DegreeCentrality_FunImgARglobalCWF.mat',
            'ROISignals_ROISignal_fALFF_FunImgARglobalCW.mat', 'ROISignals_ROISignal_ReHo_FunImgARglobalCWF.mat',
            'ROISignals_ROISignal_VMHC_FunImgARglobalCWFsymS.mat']

# feature = np.load('gretna_global_aal_90_connectivity.npy')
feature = np.load('Craddock_200_connectivity.npy')
# feature = np.load('Zalesky_980_connectivity.npy')
# print(test.shape)
# print(len(test.nonzero()[0]))

label = get_label()

# info = np.load("subinfo.npy")
# feature = np.hstack((feature,info))

data1 = get_roi_signal(filelist[0])
for x in range(1, len(filelist)):
    if x != 1:
        data1 = np.hstack((data1, get_roi_signal(filelist[x])))
    else:
        data1 = np.hstack((data1, get_roi_signal(filelist[x])[2428:]))
print(data1.shape)
'''
feature_1 = np.load('cc_white_volume.npy')
feature1 = np.load('cc_grey_volume.npy')
feature2 = np.load('cc_csf_volume.npy')
feature_1 = np.hstack((feature_1,feature1))
feature_1 = np.hstack((feature_1,feature2))
#data1 = np.hstack((data1,feature_1))
#network_feature = get_feature(2428)
#data1 = np.hstack((data1,network_feature))
#data1 = network_feature
scaler = StandardScaler()
feature_1 = scaler.fit_transform(feature_1)
'''
feature_ori = np.load('smri_mor_feature.npy')
# gretna_global_aal_90_connectivity.npy
# HO_112_connectivity.npy
feature_1 = []
for feat in feature_ori:
    iu = np.triu_indices(feat.shape[0], 1)
    # matrix = np.triu(correlation_matrix, 1).flatten()
    matrix = feat[iu]
    feature_1.append(matrix)

    # print(d)
    # print(re.split('[\\\/]', d)[3])

    # savepath = fc_matrix_dir + re.split('[\\\/]', d)[2].split('.')[0] + '_fc_matrix.txt'
    # print(savepath)
    # np.savetxt(savepath, correlation_matrix)
    # print('{} has saved!'.format(savepath))

    # print('{} has finished!'.format(d))
# print(len(test.nonzero()[0]))
feature_1 = np.array(feature_1)

# feature = np.hstack((feature,feature_1))

# feature = np.hstack((feature,feature_1))
# feature = data1
# network_feature = get_feature(2428)
# feature = np.hstack((feature,network_feature))

ex_list = select_sub()
feature = np.delete(feature, ex_list, axis=0)
#feature_1 = np.delete(feature_1, ex_list, axis=0)
label = np.delete(label, ex_list, axis=0)
'''
idx = np.arange(feature.shape[0])
np.random.shuffle(idx)
feature = feature[idx]
label = label[idx]
'''

# clf = SVC(kernel='rbf',probability=True)
# clf = RandomForestClassifier(random_state=2021)


# data = scaler.fit_transform(feature)
# data = r_z(feature)
data = feature
#data = np.hstack((feature,feature_1))
print(data.shape)
'''
pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classify', SVC(gamma='auto'))
])
N_FEATURES_OPTIONS = [100,120,150,180,200]
#C_OPTIONS = [1,10,100,1000]
#KERNEL_OPTIONS = ['rbf','poly','sigmoid']
KERNEL_OPTIONS = ['linear']

param_grid = [
    {
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        #'classify__C': C_OPTIONS,
        'classify__kernel':KERNEL_OPTIONS
    },
]
grid = GridSearchCV(pipe,cv=10,param_grid=param_grid)
grid.fit(data,label)
mean_scores = np.array(grid.cv_results_['mean_test_score'])
print(grid.best_params_)
print(grid.best_score_)
print(mean_scores)
'''

balanced_result = []
accuracy_result = []
sensitivity_result = []
specificity_result = []
auc_result = []
f1 = []
# data = PCA(n_components=200).fit_transform(data)
tn_list = []
fp_list = []
fn_list = []
tp_list = []
k = 0
for i in [42]:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
    # 十折交叉的应用

    for train_index, test_index in skf.split(data, label):

        new_label = to_categorical(label)
        X1_train, X1_test = data[train_index], data[test_index]
        X2_train, X2_test = feature_1[train_index], feature_1[test_index]
        y_train, y_test = new_label[train_index], new_label[test_index]
        bestModelSavePath = 'fold_e_mask%s/dnn_e%s_weights_best.hdf5' % (str(k), str(k))
        model = Concat_DNN_model(data.shape[1],feature_1.shape[1], 2)
        checkpoint = ModelCheckpoint(bestModelSavePath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     mode='auto')
        # selector = RFE(clf,n_features_to_select=80)
        # X_train = selector.fit_transform(X_train,y_train)
        # X_test = selector.transform(X_test)
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
        history = model.fit([X1_train,X2_train], y_train, batch_size=64, epochs=300, validation_data=([X1_test,X2_test], y_test),
                            callbacks=[checkpoint])
        model.load_weights(bestModelSavePath)
        print('history:')
        print(history.history)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Trainning acc')
        plt.plot(epochs, val_acc, 'b', label='Vaildation acc')
        plt.legend()

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Trainning loss')
        plt.plot(epochs, val_loss, 'b', label='Vaildation loss')
        plt.legend()
        # plt.show()

        result = model.evaluate([X1_test,X2_test], y_test, batch_size=128)
        print('evaluate:')
        print(result)
        y_predprob = model.predict([X1_test,X2_test])
        print('predict:')
        print(y_predprob)
        y_pred = np.where(y_predprob > 0.5, 1, 0)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        y_predprob = y_predprob[:, 1]
        # y_pred = np.argmax(y_predprob)
        print(y_pred)
        # print(y_test)
        balanced_result.append(balanced_accuracy_score(y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tp_list.append(tp)
        # fpr,tpr,threshold = metrics.roc_curve(y_test,y_predprob)
        sensitivity_result.append(tp / (tp + fn))
        specificity_result.append(tn / (fp + tn))
        accuracy_result.append(result[1])
        auc_result.append(roc_auc_score(y_test, y_predprob))
        f1.append(f1_score(y_test, y_pred, zero_division=1))
        # print(clf.score(X_test,y_test))
        # keras.clear_session()
        del model
        del history
        k += 1

print("average accuracy: " + str(np.mean(accuracy_result)))
print("var:" + str(np.var(accuracy_result)))
print("max:" + str(np.max(accuracy_result)))
print("min:" + str(np.min(accuracy_result)))
print("average balanced_accuracy: " + str(np.mean(balanced_result)))
print("var:" + str(np.var(balanced_result)))
print("max:" + str(np.max(balanced_result)))
print("min:" + str(np.min(balanced_result)))
print("average sensitivity: " + str(np.mean(sensitivity_result)))
print("var:" + str(np.var(sensitivity_result)))
print("max:" + str(np.max(sensitivity_result)))
print("min:" + str(np.min(sensitivity_result)))
print("average specificity: " + str(np.mean(specificity_result)))
print("var:" + str(np.var(specificity_result)))
print("max:" + str(np.max(specificity_result)))
print("min:" + str(np.min(specificity_result)))
print("auc accuracy: " + str(np.mean(auc_result)))
print("var:" + str(np.var(auc_result)))
print("max:" + str(np.max(auc_result)))
print("min:" + str(np.min(auc_result)))
print("average f1 score:" + str(np.mean(f1)))
print("var:" + str(np.var(f1)))
print("max:" + str(np.max(f1)))
print("min:" + str(np.min(f1)))
print("average tn:" + str(np.mean(tn_list)))
print("average fp:" + str(np.mean(fp_list)))
print("average fn:" + str(np.mean(fn_list)))
print("average tp:" + str(np.mean(tp_list)))
final_result = [(np.mean(accuracy_result), np.var(accuracy_result), np.mean(balanced_result), np.var(balanced_result),
                 np.mean(sensitivity_result), np.var(sensitivity_result),
                 np.mean(specificity_result), np.var(specificity_result), np.mean(auc_result), np.var(auc_result),
                 np.mean(f1), np.var(f1), np.mean(tn_list), np.mean(fp_list),
                 np.mean(fn_list), np.mean(tp_list))]
analyse_table = pd.DataFrame(final_result,
                             columns=['accuracy', 'acc var', 'balanced accuracy', 'balanced acc var', 'sensitivity',
                                      'sensitivity var',
                                      'specificity', 'specificity var', 'auc', 'auc var', 'f1', 'f1 var',
                                      'tn', 'fp', 'fn', 'tp'])
analyse_table.to_csv('exp_result_ml.csv')
end = time.perf_counter()
print("time: ", end - start)
