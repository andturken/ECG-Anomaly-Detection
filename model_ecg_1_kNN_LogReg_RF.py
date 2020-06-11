
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
from imblearn.metrics import classification_report_imbalanced

from joblib import dump, load
#%%




#%%

param_grid = { 'n_neighbors': [3,4,5,10,20]}
clf = KNeighborsClassifier()
clf = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
clf.fit(X_train, y_train.ravel())
# In[50]: clf.best_params_
# Out[50]: {'n_neighbors': 3}
y_test_pred = clf.predict(X_test)
print(classification_report(y_test, y_test_pred,     target_names=['N','A','L','R','V']))
#               precision    recall  f1-score   support
#
#            N       0.99      1.00      1.00     12492
#            A       0.98      0.92      0.95       524
#            L       0.99      0.99      0.99       672
#            R       0.99      1.00      0.99      1338
#            V       0.98      0.97      0.98      1373
#
#     accuracy                           0.99     16399
#    macro avg       0.99      0.97      0.98     16399
# weighted avg       0.99      0.99      0.99     16399
y_out_pred = clf.predict(X_test_out)
print(classification_report(y_test_out, y_out_pred,     target_names=['N','A','L','R','V']))

#               precision    recall  f1-score   support
#
#            N       0.85      0.98      0.91     24409
#            A       0.19      0.23      0.21       444
#            L       1.00      0.42      0.59      4073
#            R       0.33      0.04      0.07      1892
#            V       0.59      0.70      0.64      1086
#
#     accuracy                           0.83     31904
#    macro avg       0.59      0.47      0.48     31904
# weighted avg       0.82      0.83      0.80     31904


#%%
# clf = search
# print("Best parameters set found on development set:")
# print()
# print(clf.best_params_)
# print()
# print("Grid scores on development set:")
# print()
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))
# print()
#
# print("Detailed classification report:")
# print()
# print("The model is trained on the full development set.")
# print("The scores are computed on the full evaluation set.")
# print()

#%%
from sklearn.linear_model import LogisticRegression

#clf = LogisticRegression( C=100, penalty='l2', solver='saga', tol=0.01, class_weight='balanced')

param_grid = { 'C': [1, 10, 50, 100, 1000], 'penalty':['l2','none']}
clf = LogisticRegression()
clf = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
clf.fit(X_train, y_train.ravel())

print(clf.best_params_)
# {'C': 10, 'penalty': 'l2'}

y_test_pred = clf.predict(X_test)
print(classification_report(y_test, y_test_pred,     target_names=['N','A','L','R','V']))

y_out_pred = clf.predict(X_test_out)
print(classification_report(y_test_out, y_out_pred,     target_names=['N','A','L','R','V']))

clf_log_reg = clf

dump(clf_log_reg, 'Model_LogisticRegression.joblib')

# clf = load('filename.joblib')

#               precision    recall  f1-score   support
#
#            N       0.95      0.97      0.96     12492
#            A       0.73      0.66      0.69       524
#            L       0.92      0.93      0.93       672
#            R       0.86      0.85      0.86      1338
#            V       0.81      0.64      0.71      1373
#
#     accuracy                           0.92     16399
#    macro avg       0.85      0.81      0.83     16399
# weighted avg       0.92      0.92      0.92     16399
#
#               precision    recall  f1-score   support
#
#            N       0.81      0.91      0.86     24409
#            A       0.07      0.07      0.07       444
#            L       0.04      0.00      0.00      4073
#            R       0.06      0.04      0.05      1892
#            V       0.17      0.43      0.25      1086
#
#     accuracy                           0.72     31904
#    macro avg       0.23      0.29      0.25     31904

#%%
from imblearn.ensemble import BalancedRandomForestClassifier

# clf = BalancedRandomForestClassifier( n_estimators=1000, class_weight= 'balanced', random_state=0, n_jobs=-1, oob_score = True)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(classification_report_imbalanced(y_test, y_pred,     target_names=['N','A','L','R','V']))


param_grid = { 'n_estimators': [50, 100, 1000],
               'max_features': [ 'sqrt', 'log2', None],
                'class_weight': ['balanced', 'balanced_subsample']}
clf = BalancedRandomForestClassifier(oob_score = True)
clf = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
clf.fit(X_train, y_train.ravel())

print(clf.best_params_)


y_test_pred = clf.predict(X_test)
print(classification_report(y_test, y_test_pred,     target_names=['N','A','L','R','V']))

y_out_pred = clf.predict(X_test_out)
print(classification_report(y_test_out, y_out_pred,     target_names=['N','A','L','R','V']))

clf_balanced_RF = clf



dump(clf_balanced_RF, 'Model_balanced_RF.joblib')
#
# {'class_weight': 'balanced_subsample', 'max_features': 'log2', 'n_estimators': 1000}
#               precision    recall  f1-score   support
#
#            N       1.00      0.98      0.99     12492
#            A       0.81      0.95      0.87       524
#            L       0.99      0.99      0.99       672
#            R       0.99      0.99      0.99      1338
#            V       0.88      0.99      0.93      1373
#
#     accuracy                           0.98     16399
#    macro avg       0.93      0.98      0.95     16399
# weighted avg       0.98      0.98      0.98     16399
#
#               precision    recall  f1-score   support
#
#            N       0.89      0.92      0.90     24409
#            A       0.08      0.44      0.14       444
#            L       1.00      0.38      0.55      4073
#            R       0.52      0.04      0.07      1892
#            V       0.38      0.86      0.53      1086
#
#     accuracy                           0.79     31904
#    macro avg       0.57      0.53      0.44     31904
# weighted avg       0.85      0.79      0.78     31904
#
