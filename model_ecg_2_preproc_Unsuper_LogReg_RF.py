
#%%

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os


#%%
import pickle

from sklearn.metrics import classification_report

with open('data_ecg.pickle', 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    data_dict = pickle.load(f)

for key in data_dict.keys():
    globals()[key] = data_dict[key]

 #   data_all.shape
#%%

# List all ECG type annotations, and their frequencies of occurrence
# Will use five most common types
a = np.unique(annotations_all)
L =[]
for aa in a:
    L.append((aa, sum(annotations_all==aa)))
    print(aa, sum(annotations_all==aa))
#%%
# flag potential ecg artifacts: excessive variability and large negative deflections
std = np.std(data_all, axis=1)
std_th = std > np.mean(std) + 4*np.std(std)
amp_max_neg = np.min(data_all, axis=1)
amp_th = amp_max_neg < np.mean(amp_max_neg) - 4 * np.std(amp_max_neg)
# amp_th.sum(), std_th.sum(), np.logical_and( amp_th , std_th).sum()

ix = np.logical_and(~std_th, ~amp_th)
data = data_all[ix,:]
annotations = annotations_all[ix]
onsets = onsets_all[ix]
pt_codes = pt_codes_all[ix]

#%%
# heart beat types
# N = Normal
# focus on A, L, R, V five most common heart beat anomaly types

tN = np.where(annotations=='N')[0]
tA = np.where(annotations=='A')[0]
tL = np.where(annotations=='L')[0]
tR = np.where(annotations=='R')[0]
tV = np.where(annotations=='V')[0]

#indices for normal ecg and five most common abnormal ecg types
ix = np.concatenate([tN, tA, tL, tR, tV])


data = data[ix,:]
annotations = annotations[ix]
onsets = onsets_all[ix]
pt_codes = pt_codes_all[ix]

labels_str = annotations
labels = np.zeros((len(labels_str), 1))
# for i, lab in enumerate(np.unique(labels_str)):
#     labels[labels_str==lab] = i
labels[labels_str=='N']=0
labels[labels_str=='A']=1
labels[labels_str=='L']=2
labels[labels_str=='R']=3
labels[labels_str=='V']=4


#%%

# Training  and test datasets:

# ECG data from 30 patients used for training (further split during cross-validation
# and in-sample testing (data from same patients used for both training and testing performance)
# ECG data from furhter 14 patients used for out-sample testing,
# to assess model performance on entirely new data

# pts_1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
# pts_2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
pts_1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 121, 123, 200,
         201, 203, 205, 207, 208, 209, 215, 220, 223, 230, 228, 231, 232, 233, 234]
pts_2 = [100, 103, 105, 111, 113, 117, 202, 210, 212, 213, 214, 219, 221, 222]

ix_pts_1 = np.where( np.isin(pt_codes , pts_1))[0]
ix_pts_2 = np.where( np.isin(pt_codes , pts_2))[0]

data_pts_1, labels_pts_1, onsets_pts_1, pt_codes_pts_1 = \
    data[ix_pts_1,:], labels[ix_pts_1], onsets[ix_pts_1], pt_codes[ix_pts_1]

data_pts_2, labels_pts_2, onsets_pts_2, pt_codes_pts_2 = \
    data[ix_pts_2,:], labels[ix_pts_2], onsets[ix_pts_2], pt_codes[ix_pts_2]


ix_train = np.where( np.isin(pt_codes , pts_1))[0]
ix_test_out  = np.where( np.isin(pt_codes , pts_2))[0]

data_train, labels_train, onsets_train, pt_codes_train = \
    data[ix_train,:], labels[ix_train], onsets[ix_train], pt_codes[ix_train]

data_test_out, labels_test_out, onsets_test_out, pt_codes_test_out = \
    data[ix_test_out,:], labels[ix_test_out], onsets[ix_test_out], pt_codes[ix_test_out]


iX = list(range(data_train.shape[0]))
X = data_train
y = labels_train.ravel()

iX_train,  iX_test, y_train, y_test = \
    train_test_split(iX, y, test_size=0.25, stratify=y, random_state=0)

X_train, X_test = X[iX_train], X[iX_test]

X_test_out = data_test_out
y_test_out = labels_test_out

#%%
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#%%
from sklearn.pipeline import Pipeline

from sklearn.decomposition import MiniBatchSparsePCA

from sklearn.decomposition import MiniBatchDictionaryLearning

from sklearn.decomposition import FastICA

#%%
mbsp = MiniBatchSparsePCA()
mbdl = MiniBatchDictionaryLearning()
fica = FastICA()

logreg = LogisticRegression(class_weight='balanced', solver='sag', max_iter=5000)
rf = BalancedRandomForestClassifier(class_weight='balanced')



p_mbsp = { 'spca__n_components' :  [5 , 15 , 25] }
p_mbdl = { 'dl__n_components' :  [5 , 15 , 25] }
p_fica = { 'ica__n_components' :  [5  ,15 , 25] }

p_mbsp = { 'spca__n_components' :  [ 25] }
p_mbdl = { 'dl__n_components' :25] }
p_fica = { 'ica__n_components' :  [ 25] }


p_logreg = { 'logreg__C': [ 1] }
p_rf = { 'rf__n_estimators': [ 1000],
               'rf__max_features': [ 'sqrt'] }


s_spca = ( 'spca', mbsp, p_mbsp)
s_dl = ('dl', mbdl, p_mbdl)
s_ica = ('ica', fica, p_fica)

s_lr = ('logreg', logreg, p_logreg)
s_rf = ('rf', rf, p_rf)

# pipe = Pipeline(steps=[('spca', mbsp), ('logistic', logreg)])


#%%

pipe_names = []
params = []
pipes = []
grids = []

for s1 in [s_spca, s_dl, s_ica]:
    for s2 in [s_lr, s_rf]:
        nm = s1[0] + '_' + s2[0]
        pipe_names.append(nm)
        prm = s1[2].copy()
        prm.update(s2[2])
        params.append(prm)
        pipe= Pipeline(steps=[s1[:2], s2[:2]])
        pipes.append(pipe)
        grid = GridSearchCV(pipe, prm, cv=5, n_jobs=-1)
        grids.append(grid)
        print(nm)
        print(pipe)
        print(prm)
        print(grid)

#%%
# Test sample classifier in grids list
# print(grids[-1])
# GridSearchCV(cv=None, error_score=nan,
#              estimator=Pipeline(memory=None,
#                                 steps=[('spca',
#                                         MiniBatchSparsePCA(alpha=1,
#                                                            batch_size=3,
#                                                            callback=None,
#                                                            method='lars',
#                                                            n_components=None,
#                                                            n_iter=100,
#                                                            n_jobs=None,
#                                                            normalize_components='deprecated',
#                                                            random_state=None,
#                                                            ridge_alpha=0.01,
#                                                            shuffle=True,
#                                                            verbose=False)),
#                                        ('rf',
#                                         BalancedRandomForestClassifier(bootstrap=True,
#                                                                        ccp_alph...
#                                                                        n_estimators=100,
#                                                                        n_jobs=None,
#                                                                        oob_score=False,
#                                                                        random_state=None,
#                                                                        replacement=False,
#                                                                        sampling_strategy='auto',
#                                                                        verbose=0,
#                                                                        warm_start=False))],
#                                 verbose=False),
#              iid='deprecated', n_jobs=-1,
#              param_grid={'rf__max_features': ['sqrt'],
#                          'rf__n_estimators': [100],
#                          'spca__n_components': [50]},
#              pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
#              scoring=None, verbose=0)
#%%

for i, grid in enumerate(grids):
    print(pipe_names[i])
    # try:
    #     grid.fit(X_train, y_train)
    #     print(grid.best_params)
    # except:
    #     pass
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_score_)

#%%
# Print test scores for best model in each pipeline
# Save each model to disk using joblib

from joblib import dump, load

for i in range(len(grids)):
    print(i)
    clf = grids[i]
    try:
        print(pipe_names[i])
        print(clf.best_params_)
        print(clf.best_score_)

        y_test_pred = clf.predict(X_test)
        print(classification_report(y_test, y_test_pred,     target_names=['N','A','L','R','V']))

        y_out_pred = clf.predict(X_test_out)
        print(classification_report(y_test_out, y_out_pred,     target_names=['N','A','L','R','V']))

        # dump(clf_balanced_RF, 'Model_' + pipe_names[i]  + '.joblib')
    except Exception as e:
        print(e)


#%%



#%%

#%%
