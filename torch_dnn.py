import os

from torch.utils.data import random_split
from torch_geometric.data import DataLoader

import torch
from torch_geometric.data import dataset
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp,GlobalAttention as GA
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
import scipy.sparse as sp
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
from construct_graph_for_pyg import RESTDataset
from construct_smri_graph_for_pyg import REST_smri_Dataset
from get_label import get_label
from select_sub import select_sub
import pandas as pd
'''
class Net(torch.nn.Module):
    def __init__(self,num_classes):
        super(Net,self).__init__()

        self.num_classes = num_classes
        self.dl1 = torch.nn.Linear(19900,64)
        self.dl2 = torch.nn.Linear(64,16)
        self.dl3 = torch.nn.Linear(19900,64)
        self.dl4 = torch.nn.Linear(64, 16)
        self.dl5 = torch.nn.Linear(32, num_classes)
        self.b1 = torch.nn.BatchNorm1d(num_features=64)
        self.b2 = torch.nn.BatchNorm1d(num_features=64)
        self.b3 = torch.nn.BatchNorm1d(num_features=16)
        self.b4 = torch.nn.BatchNorm1d(num_features=16)
        torch.nn.init.kaiming_normal_(self.dl1.weight, a=1)
        torch.nn.init.kaiming_normal_(self.dl2.weight, a=1)
        torch.nn.init.kaiming_normal_(self.dl3.weight, a=1)
        torch.nn.init.kaiming_normal_(self.dl4.weight, a=1)
        torch.nn.init.kaiming_normal_(self.dl5.weight, a=1)
    def forward(self,data3,data4):
        data3 = self.dl1(data3)
        data3 = self.b1(data3)
        data3 = F.leaky_relu(data3)
        data3 = F.dropout(data3,0.5, training=self.training)

        data4 = self.dl3(data4)
        data4 = self.b2(data4)
        data4 = F.leaky_relu(data4)
        data3 = F.dropout(data3, 0.5, training=self.training)

        data3 = self.dl2(data3)
        data3 = self.b3(data3)
        data3 = F.leaky_relu(data3)
        data3 = F.dropout(data3,0.5, training=self.training)

        data4 = self.dl4(data4)
        data4 = self.b4(data4)
        data4 = F.leaky_relu(data4)
        data4 = F.dropout(data4,0.5, training=self.training)

        x = torch.cat([data3,data4],dim=1)
        x = self.dl5(x)

        return F.softmax(x)
'''
class Net(torch.nn.Module):
    def __init__(self,num_classes):
        super(Net,self).__init__()

        self.num_classes = num_classes
        self.dl1 = torch.nn.Linear(39800,128)
        self.dl2 = torch.nn.Linear(128,32)
        self.dl5 = torch.nn.Linear(32, num_classes)
        self.b1 = torch.nn.BatchNorm1d(num_features=128)
        self.b2 = torch.nn.BatchNorm1d(num_features=32)
        torch.nn.init.kaiming_normal_(self.dl1.weight, a=1)
        torch.nn.init.kaiming_normal_(self.dl2.weight, a=1)
        torch.nn.init.kaiming_normal_(self.dl5.weight, a=1)
    def forward(self,data):
        data = self.dl1(data)
        data = self.b1(data)
        data = F.leaky_relu(data)
        data = F.dropout(data,0.5, training=self.training)

        data = self.dl2(data)
        data = self.b2(data)
        data = F.leaky_relu(data)
        data = F.dropout(data,0.5, training=self.training)
        x = self.dl5(data)

        return F.softmax(x)

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, fmri_vector,smri_vector,label):
        self.fmriv = fmri_vector
        self.smriv = smri_vector
        self.label = label

    def __len__(self):
        return len(self.fmriv)

    def __getitem__(self, idx):
        #data = (self.csv_data[idx], self.txt_data[idx])
        return self.fmriv[idx],self.smriv[idx],self.label[idx]

class SingleModalDataset(torch.utils.data.Dataset):
    def __init__(self, vector,label):
        self.vec = vector
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        #data = (self.csv_data[idx], self.txt_data[idx])
        return self.vec[idx],self.label[idx]

device = 'cpu'
#num_features = 116
seed = 777
batch_size = 64
lr = 0.001
weight_decay = 0.0001
nhid = 32
pooling_ratio = 0.5
dropout_ratio = 0.6
epochs = 300
patience = 30
#pooling_layer_type = 'GCNConv'
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    device = 'cuda:0'
#X = get_conn("extracted/GretnaSFCMatrixZ")
y = get_label()
Y = np.reshape(y,[-1,1])
X = Y
ex_list = select_sub()
X = np.delete(X,ex_list,axis=0)
Y = np.delete(Y,ex_list,axis=0)
features = X

num_classes = 2
num_features = 200
loss_func = torch.nn.CrossEntropyLoss()
'''
def model_test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    predlist = []
    predprob = []
    gt = []
    for data1,data2,label in loader:
        data1 = data1.to(device)
        data2 = data2.to(device)
        label = label.to(device)
        out = model(data1,data2)
        pred = out.max(dim=1)[1]
        prednp = out.max(dim=1)[1].cpu().numpy()
        predprob.append(out.data.cpu().numpy()[:,1])
        predlist.append(prednp[0])
        gt.append(label.cpu().numpy()[0])
        correct += pred.eq(label).sum().item()
        loss += loss_func(out,label).item()
    return correct / len(loader.dataset),loss / len(loader.dataset),predlist,predprob,gt
'''
def model_test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    predlist = []
    predprob = []
    gt = []
    for data1,label in loader:
        data1 = data1.to(device)
        label = label.to(device)
        out = model(data1)
        pred = out.max(dim=1)[1]
        prednp = out.max(dim=1)[1].cpu().numpy()
        predprob.append(out.data.cpu().numpy()[:,1])
        predlist.append(prednp[0])
        gt.append(label.cpu().numpy()[0])
        correct += pred.eq(label).sum().item()
        loss += loss_func(out,label).item()
    return correct / len(loader.dataset),loss / len(loader.dataset),predlist,predprob,gt
balanced_result = []
accuracy_result = []
sensitivity_result = []
specificity_result = []
auc_result = []
f1 = []
tn_list = []
fp_list = []
fn_list = []
tp_list = []

feature = np.load('Craddock_200_connectivity.npy')
feature_ori = np.load('cc_smriz.npy')
feature_1 = []
for feat in feature_ori:
    iu = np.triu_indices(feat.shape[0], 1)
    matrix = feat[iu]
    feature_1.append(matrix)
feature_1 = np.array(feature_1)
label = get_label()

ex_list = select_sub()
feature = np.delete(feature, ex_list, axis=0)
feature_1 = np.delete(feature_1, ex_list, axis=0)
feature = np.concatenate((feature,feature_1),axis=1)
feature = torch.from_numpy(feature)
feature = feature.to(torch.float32)
#feature_1 = torch.from_numpy(feature_1)
#feature_1 = feature_1.to(torch.float32)
label = np.delete(label,ex_list,axis=0)
label = torch.from_numpy(label)
label = label.long()
'''
model = Net(num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# print(model)
min_loss = 1e10
patience = 0
max_acc = 0
for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader):
            #print(data)
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            loss = loss_func(out, data.y)
            print("Training loss:{}".format(loss.item()))
            loss.backward()
            optimizer.step()

        val_acc,val_loss,_,_,_ = model_test(model,test_loader)
        print("Epoch {}".format(epoch))
        print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
        if val_acc > max_acc:
            torch.save(model.state_dict(),'gat_latest.pth')
            print("Model saved at epoch{}".format(epoch))
            max_acc = val_acc
            patience = 0
        else:
            patience += 1
        if patience > patience:
            break


model = Net(num_classes).to(device)
model.load_state_dict(torch.load('gat_latest.pth'))
test_acc,test_loss,predlist,predprob,gt  = model_test(model,test_loader)
'''
#print(dataset_multi[0])
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
k = 0
#dataset_fmri = RESTDataset('dataset/rest-meta-mdd')
#dataset_smri = REST_smri_Dataset('dataset/rest-meta-mdd-smri')
for train_index, test_index in skf.split(features, Y):

    #dataset_multi = MultiModalDataset(dataset_fmri, dataset_smri)
    #training_set = MultiModalDataset(feature[train_index],feature_1[train_index],label[train_index])
    training_set = SingleModalDataset(feature[train_index],label[train_index])
    #training_set = dataset_multi
    #test_set = dataset_multi[list(test_index)]
    test_set = SingleModalDataset(feature[test_index],label[test_index])
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
    model = Net(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #print(model)

    min_loss = 1e10
    patience = 0
    max_acc = 0
    for epoch in range(epochs):
        model.train()
        #print(train_loader)
        for i,data in enumerate(train_loader):
            data1,gt1 = data
            optimizer.zero_grad()
            data1 = data1.to(device)
            gt1 = gt1.to(device)
            out = model(data1)
            loss = loss_func(out, gt1)
            print("Training loss:{}".format(loss.item()))
            loss.backward()
            optimizer.step()

        val_acc,val_loss,_,_,_ = model_test(model,test_loader)
        print("Current round:" + str(k))
        print("Epoch {}".format(epoch))
        print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
        print("Current max val acc:"+str(max_acc))
        if val_acc > max_acc:
            torch.save(model.state_dict(),'/data/hym_code/gat_latest'+str(k)+'.pth')
            print("Model saved at epoch{}".format(epoch))
            max_acc = val_acc
            patience = 0
        else:
            patience += 1
        if patience > patience:
            break
    del model

    model = Net(num_classes).to(device)
    model.load_state_dict(torch.load('/data/hym_code/gat_latest'+str(k)+'.pth'))
    test_acc,test_loss,predlist,predprob,gt  = model_test(model,test_loader)
    print("Test accuarcy:{}".format(test_acc))
    y_pred = np.array(predlist)
    y_predprob = np.array(predprob)
    y_test = np.array(gt)
    # print(y_test)
    # print(y_pred)
    balanced_result.append(balanced_accuracy_score(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)
    tp_list.append(tp)
    # fpr,tpr,threshold = metrics.roc_curve(y_test,y_predprob)
    sensitivity_result.append(tp / (tp + fn))
    specificity_result.append(tn / (fp + tn))
    accuracy_result.append(accuracy_score(y_test.data, y_pred))
    auc_result.append(roc_auc_score(y_test.data, y_predprob))
    f1.append(f1_score(y_test, y_pred, zero_division=1))
    # print(clf.score(X_test,y_test))
    # keras.clear_session()
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
final_result = [(np.mean(accuracy_result),np.var(accuracy_result),np.mean(balanced_result),np.var(balanced_result),np.mean(sensitivity_result),np.var(sensitivity_result),
          np.mean(specificity_result),np.var(specificity_result),np.mean(auc_result),np.var(auc_result),np.mean(f1),np.var(f1),np.mean(tn_list),np.mean(fp_list),
          np.mean(fn_list),np.mean(tp_list))]
analyse_table = pd.DataFrame(final_result, columns=['accuracy','acc var', 'balanced accuracy', 'balanced acc var', 'sensitivity', 'sensitivity var',
                                                   'specificity', 'specificity var','auc', 'auc var','f1', 'f1 var',
                                                   'tn', 'fp','fn','tp'])
analyse_table.to_csv('/data/hym_code/exp_result_ml.csv')