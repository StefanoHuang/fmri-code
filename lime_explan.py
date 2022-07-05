import os
import re

import keras
import scipy.io as sio
import numpy as np
import pandas as pd
import nilearn
import sklearn.metrics
from nilearn import masking,plotting, image, datasets,input_data
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
from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
import shap

import lime.lime_tabular
from get_label import get_label
from read_other import get_roi_signal
from read_network import get_feature
from model import DNN_model,DNN_seq_model
from select_sub import select_sub





filelist = ['ALFF_FunImgARCW_entireROISignals.mat','DegreeCentrality_FunImgARCWF_entireROISignals.mat',
            'fALFF_FunImgARCW_entireROISignals.mat','ReHo_FunImgARCWF_entireROISignals.mat','VMHC_FunImgARCWFsymS_entireROISignals.mat']

feature = np.load('gretna_global_aal_90_connectivity.npy')
#print(test.shape)
#print(len(test.nonzero()[0]))

label = get_label()

#info = np.load("subinfo.npy")
#feature = np.hstack((feature,info))

data1 = get_roi_signal(filelist[0])
for x in range(1,len(filelist)):
    if x != 1:
        data1 = np.hstack((data1,get_roi_signal(filelist[x])))
    else:
        data1 = np.hstack((data1, get_roi_signal(filelist[x])[2428:]))
print(data1.shape)
network_feature = get_feature(2428)
data1 = np.hstack((data1,network_feature))
#data1 = network_feature
scaler = StandardScaler()
data1 = scaler.fit_transform(data1)
feature = np.hstack((feature,data1))
#feature = data1
#network_feature = get_feature(2428)
#feature = np.hstack((feature,network_feature))

ex_list = select_sub()
feature = np.delete(feature,ex_list,axis=0)
label = np.delete(label,ex_list,axis=0)
'''
idx = np.arange(feature.shape[0])
np.random.shuffle(idx)
feature = feature[idx]
label = label[idx]
'''

#clf = SVC(kernel='rbf',probability=True)
#clf = RandomForestClassifier(random_state=2021)


scaler = StandardScaler()
#data = scaler.fit_transform(feature)
#data = r_z(feature)
data = feature
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
feat_name = []
for i in range(data.shape[1]):
    feat_name.append("feat"+str(i))
class_name = ['HC','MDD']

balanced_result = []
accuracy_result = []
sensitivity_result = []
specificity_result = []
auc_result = []
f1 = []
#data = PCA(n_components=200).fit_transform(data)
tn_list = []
fp_list = []
fn_list = []
tp_list = []
final_exp =[]
for i in [42]:
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=i)
    # 十折交叉的应用

    for train_index, test_index in skf.split(data, label):
        k = 0

        new_label = to_categorical(label)
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = new_label[train_index], new_label[test_index]
        bestModelSavePath = 'fold_e_mask%s/dnn_e%s_weights_best.hdf5' % (str(k), str(k))
        model = DNN_seq_model(data.shape[1], 2)



        checkpoint = ModelCheckpoint(bestModelSavePath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        # selector = RFE(clf,n_features_to_select=80)
        # X_train = selector.fit_transform(X_train,y_train)
        # X_test = selector.transform(X_test)
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=64, epochs=200,validation_data=(X_test,y_test), callbacks=[checkpoint])
        model.load_weights(bestModelSavePath)
        print('history:')
        print(history.history)
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)


        result = model.evaluate(X_test, y_test, batch_size=128)
        print('evaluate:')
        print(result)
        y_predprob = model.predict(X_test)
        print('predict:')
        print(y_predprob)
        y_pred = np.where(y_predprob>0.5,1,0)
        y_pred = np.argmax(y_pred,axis=1)
        y_test = np.argmax(y_test, axis=1)
        y_predprob = y_predprob[:,1]
        #y_pred = np.argmax(y_predprob)
        print(y_pred)
        #print(y_test)
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
        #keras.clear_session()
        start = time.perf_counter()
        print(X_train.shape)
        explainer = shap.KernelExplainer(model = model.predict, data = X_train[:200], link = "identity")
        shap_value = explainer.shap_values(X_test,nsamples=1000)
        final_exp.append(shap_value[1])
        #shap.force_plot(explainer.expected_value, shap_value, X_display.iloc[299, :])
        #shap.summary_plot(shap_value, X_test)
        end = time.perf_counter()
        print("time: ", end - start)
        del model
        del history
        k = k + 1

final_exp = np.array(final_exp)
np.save("shap_exp_result",final_exp)
print(final_exp.shape)
print("average accuracy: "+str(np.mean(accuracy_result)))
print("var:"+str(np.var(accuracy_result)))
print("max:"+str(np.max(accuracy_result)))
print("min:"+str(np.min(accuracy_result)))
print("average balanced_accuracy: "+str(np.mean(balanced_result)))
print("var:"+str(np.var(balanced_result)))
print("max:"+str(np.max(balanced_result)))
print("min:"+str(np.min(balanced_result)))
print("average sensitivity: "+str(np.mean(sensitivity_result)))
print("var:"+str(np.var(sensitivity_result)))
print("max:"+str(np.max(sensitivity_result)))
print("min:"+str(np.min(sensitivity_result)))
print("average specificity: "+str(np.mean(specificity_result)))
print("var:"+str(np.var(specificity_result)))
print("max:"+str(np.max(specificity_result)))
print("min:"+str(np.min(specificity_result)))
print("auc accuracy: "+str(np.mean(auc_result)))
print("var:"+str(np.var(auc_result)))
print("max:"+str(np.max(auc_result)))
print("min:"+str(np.min(auc_result)))
print("average f1 score:"+str(np.mean(f1)))
print("var:"+str(np.var(f1)))
print("max:"+str(np.max(f1)))
print("min:"+str(np.min(f1)))
print("average tn:"+str(np.mean(tn_list)))
print("average fp:"+str(np.mean(fp_list)))
print("average fn:"+str(np.mean(fn_list)))
print("average tp:"+str(np.mean(tp_list)))


