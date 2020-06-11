Models produced by executing Model_ecg_kNN_LogReg_RF.ipynb  

Model_kNN.joblib 
Model_LogReg.joblib
Model_balanced_RF.joblib  

each model can be loaded and fitter as a sklearn classfier object using

from joblib import dump, load 
clf = load('Model_xxx.joblib')
pred = clf.fit(X_train, y_train)
