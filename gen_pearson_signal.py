import math
import os

import scipy.io as sio
import numpy as np
''''''
def get_roi_signal(filename):
    signal = sio.loadmat(filename)['ROISignals']
    return signal

filelist = ['ROISignals_ROISignal_ALFF_FunImgARglobalCW.mat','ROISignals_ROISignal_DegreeCentrality_FunImgARglobalCWF.mat',
            'ROISignals_ROISignal_fALFF_FunImgARglobalCW.mat','ROISignals_ROISignal_ReHo_FunImgARglobalCWF.mat','ROISignals_ROISignal_VMHC_FunImgARglobalCWFsymS.mat']
data1 = get_roi_signal(filelist[0])[:,:,np.newaxis]
for x in range(1,len(filelist)):
    if x != 1:
        temp = get_roi_signal(filelist[x])[:,:,np.newaxis]
        data1 = np.concatenate((data1,temp),axis=2)
    else:
        temp = get_roi_signal(filelist[x])[2428:, :, np.newaxis]
        data1 = np.concatenate((data1,temp),axis=2)
print(data1.shape)
#print(white.shape)
#X = get_conn("extracted/GretnaSFCMatrixZ")
#print(X.shape)
#result = np.corrcoef((feat[0][30]),feat[0][89])
#print(result)
adjacency = []
adjacency_z = []
count = 0
for subject in data1:
    count += 1
    print("for the subject of:"+str(count))
    feature = np.eye(200)
    #print(feature)
    for i in range(200):
        for j in range(200):
            #print(subject[i])
            #print(np.corrcoef(subject[i],subject[j]))
            #print(feature[i][j])
            #print(subject[i])
            feature[i][j] = np.corrcoef(subject[i],subject[j])[0][1]
    feature = (feature+feature.T)/2
    for i in range(200):
        for j in range(200):
            if math.isnan(feature[i][j]):
                feature[i][j] = 0
            if i == j:
                feature[i][j] = 0
            if feature[i][j] >= 1:
                feature[i][j] = 1-1e-16
    #print(feature)
    adjacency.append(feature)

    for i in range(200):
        for j in range(200):
            z = feature[i][j]
            feature[i][j] = (0.5*math.log((1 + z)/(1 - z)))
            if feature[i][j] > 20:
                print(feature[i][j])
    #print(feature)
    adjacency_z.append(feature)

    del feature

np.save("cc_signalr.npy",np.array(adjacency))
print(np.array(adjacency).shape)
np.save("cc_signalz.npy",np.array(adjacency_z))
print(np.array(adjacency_z).shape)
#adjs = np.load("smriz.npy")
#print(adjs.shape)
#adjacency_z = np.load('smrir.npy')
#print(np.array(adjacency_z).shape)