import numpy as np
import pandas as pd
import math
from sklearn.metrics import f1_score
import scipy


def compute_EXP_F1(pred,target):
    pred_labels = []
    pred = np.array(pred)
    target = np.array(target)
    for i in range(pred.shape[0]):
        l = np.argmax(pred[i])
        pred_labels.append(l)
    F1s = f1_score(target,pred_labels,average=None)
    macro_f1 = np.mean(F1s)
    return F1s,macro_f1


def compute_AU_F1(pred,label):
    pred = np.array(pred)
    label = np.array(label)
    AU_targets = [[] for i in range(12)]
    AU_preds = [[] for i in range(12)]
    F1s = []
    for i in range(pred.shape[0]):
        for j in range(12):
            p = pred[i,j]
            if p>=0.5:
                AU_preds[j].append(1)
            else:
                AU_preds[j].append(0)
            AU_targets[j].append(label[i,j])
    
    for i in range(12):
        F1s.append(f1_score(AU_targets[i], AU_preds[i]))

    F1s = np.array(F1s)
    F1_mean = np.mean(F1s)
    return F1s, F1_mean


def PCC(x,y):
    x = np.array(x)
    y = np.array(y)
    x[x>1] = 1
    x[x<0] = 0
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    pcc = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    return pcc

def compute_emo_PCC(x,y):
    x = np.array(x)
    y = np.array(y)
    pccs = []
    for i in range(7):
        p = PCC(x[:,i],y[:,i])
        pccs.append(p)
    pccs = np.array(pccs)
    mean_pcc = np.mean(pccs)
    return pccs, mean_pcc

def CCC_score(x, y):
    x = np.array(x)
    y = np.array(y)
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

def compute_VA_CCC(x,y):
    x = np.array(x)
    y = np.array(y)
    x[x>1] = 1
    x[x<-1] = -1
    print(x.shape,y.shape)
    ccc1 = CCC_score(x[:,0],y[:,0])
    ccc2 = CCC_score(x[:,1],y[:,1])
    return ccc1,ccc2


    