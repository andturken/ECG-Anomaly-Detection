# ECG-Anomaly-Detection
Machines Learning for Monitoring Abnormal Heart Beats on Electrocardiogram (ECG) Recordings


# Problem Statement
Monitoring heart health is an important medical problem: abnormalities in heart activity can be warning signs of serious adverse medical events such as heart attack and stroke. Heart activity is monitored in clinical settings with the electrocardiogram (ECG), as device which records and digitizes electrical signals from the heart. Internet-of-things (IoT) devices, such as smartwatches, can now record ECG signals. These can be transmitted to a computer in order to detect and classify abnormal heartbeats in order to warn users as well as their heath care providers. State-of-the-art medical ECG systems acquire high-temporal-resolution signals from several channels at once, and process these signals on dedicated hardware. ECG recordings from smartwatches currently provide data from a single channel, as lower temporal resolution, and possibly with lower signal quality. ECG data from a smartwatch might be processed on a  device with low power computational power, such as a smartphone, or a user's laptop. Thus, applying standard machine learning approaches to detect and classify abnormal heartbeats from low resolution ECG data is a worthwhile goal for improving new IoT-based ECG recording system.  For this purpose, the present project implements and assesses Python-based machine learning solutions to learn to classify ECG heart beat recordings.


# Data sources:
## MIT-BIH Arrhythmia Database (Requires creating an account to access the data)
https://physionet.org/content/mitdb/1.0.0/ or https://console.cloud.google.com/storage/browser/mitdb-1.0.0.physionet.org
Clinical ECG recordings (30 min, dual-lead, 125 Hz) from 48 adults, 23 with rare heart conditions
109,000 manually segmented, expert-labeled heart beat samples
Standard ECG data format (.dat) with expert annotations (.atr, text labels)

WFDB Python library (https://github.com/MIT-LCP/wfdb-python) is required to read ECG data in .dat / .atr format

Alternatively, same data, in csv format, are available from [Kaggle](https://www.kaggle.com/mondejar/mitbih-database)   
(requires Kaggle account for access)


# In order to replicate the data analyses steps described in the accompanying [presentation](https://github.com/andturken/ECG-Anomaly-Detection/blob/master/Presentation%20-%20ECG%20Abnormality%20Detection.pdf): 

## To clone this repository
git clone https://github.com/andturken/ECG-Anomaly-Detection.git
cd ECG-Anomaly-Detection

Depends on: Python 3.7, scipy, pandas, numpy, matplotlib, sklearn (0.23.1), imbalanced-learn


## Pre-processing ECG data / Exploratory data analysis (EDA)

The script [download_preprocess_ecg.py](https://github.com/andturken/ECG-Anomaly-Detection/blob/master/download_preprocess_ecg.py) will: 
1-) Read MIT-BIH ECG data in CSV format into memory 
2-) Extract ECG segments corresponding to individual heart beats (~2500 heart beats per patients, consistent with 80 beats per min)
3-) Apply linear drift and pre-peak baseline corrections, alignment of ECG peak latency across samples, amplitude scaling to 100
4-) Reject ECG segments with std across time points > 3 s.d., and raw peak amp > 3 s.d., as these are likely recording artifacts
5-) Only normal heart beats (>90% of all ECG data) and the four most common abnormal heart beat ECG types are used
6-) The resulting numpy arrays are ready for modeling: 
    data_all: rows - individual ECG segments, corresponding to single heart beats; columns: time points along the ECG time course
    annotations_all - expert-provided labels, indicating whether a heart beat is normal ('N'), or the type of abnormality ('A', 'V'...)
    patients_all - for each heart beat segment (rows of data_all matrix), the id code for the corresponding patient
    onsets_all - for each heart beat segment, its onset latency relative to the corresponding continuous ECG data file

The resulting pre-processed ECG dataset is saved in the [preprocessed_ecg folder](https://github.com/andturken/ECG-Anomaly-Detection/blob/master/preprocessed_ecg/shelve_ECG_segments_mitbih.out.dir)
 
The script [load_preprocessed_ecg.py](https://github.com/andturken/ECG-Anomaly-Detection/blob/master/load_ecg_data_matrix.py) contains functions that will load preprocessed ECG data matrices into python memory, and for implementing threshold-based artifact rejection

## Key observations from EDA
Appropriate pre-processing is critical before further modeling the data as:
- Continuous ECG data shows low frequency amplitude fluctuations
- There are occasional high amplitude spikes
- There is variability in the peak latency as well as individual timing and amplitude of ECG components from heart beat to heart beat
- There are occasional bursts of high frequency noise
- Adjusting for baseline drifts, aligning ECG peaks across heart beat segments, rejecting artifactual segments and scaling all heart    beat segments to constant amplitude ensures that distance and similarity measures can be correctly computed for optimal classification
- Due to high class imbalance between normal and abnormal heart beats, modeling and model evaluation metrics have to take this class probability imbalance into account

![Continuous ECG](https://github.com/andturken/ECG-Anomaly-Detection/blob/master/images/Continuous_ECG_sample.jpeg)      ![Single ECG segment](https://github.com/andturken/ECG-Anomaly-Detection/blob/master/images/ECG_segments_standardized.jpg)

Left: Example of continuous ECG time series (containing several normal heart beats) exhibiting slow baseline amplitude fluctuations, as well as differences in signal gain across time, causing heart beats in the latter half of the tine series to appear to have larger peak amplitudes. While a diagnostician who is visually monitoring for abnormal heart beats mentally takes variable in recording quality into account, an automated computer algorithm can mistake such changes due to the recording environment for genuine changes in heart activity, confounding similarity and distance computations when comparing single heart beats based on ECG signal amplitude at each time point. Also note that the time course of short ECG changes surrounding each ECG peak varies from beat to beat. 

Right: ECG segments corresponding to individual heart beats become better comparable for computational analysis, after baseline, signal gain and peak latency variability is controlled for. EDA showed that without such corrections, classification accuracy can be reduced by up to 20%.

## Modeling - Classification of heartbeats into Normal vs one of four possible abnormality types

The modeling steps are illustrated in the accompanying Jupyter notebook, [Model_ecg_kNN_LogReg_RF.ipynb](https://github.com/andturken/ECG-Anomaly-Detection/blob/master/Model_ecg_kNN_LogReg_RF.ipynb)

The python script [model_ecg.py](https://github.com/andturken/ECG-Anomaly-Detection/blob/master/model_ecg.py) can be used to execute all modeling steps from the console, and to save best performing models as sklearn joblib files. Saved models can be loaded into memory and executed with, e.g.:

from sklearn import joblib
clf = joblib.load('Model_LogisticRegression.joblib')
predicted_abnormality_type = clf.predict(ECG_matrix__test)
(0=Normal; 1,2,3,4: one of four possible abnormal heart beat types

### Metric for evaluating model performance: 
Five-fold cross-validated F1, weighted across five classes (F1 for one-vs-rest classification for each class), inversely weighted by the corresponding class probability

### Rationale for choosing the weighted F1 metric: 
1) There is high imbalance across classes; 2) Missing abnormal heart beats, which might predict disease, and declaring abnormalities when there are none (potentially causing needless distress),are both undesirable in a clinical application. F1 captures the appropriate balance between precision and recall

Note: While for purposes of this analysis, the focus is on the classification of ECG segments for individual heart beats, in any diagnostic clinical application, assessment of abnormal heart activity would rely on several minutes worth of ECG  recordings, corresponding to hundreds of heart beats. Therefore, any metric based on how well single heart beats are classified can be can be considered conservative.

### Overview of modeling steps:
1- Patient datasets were split into two groups, n1=21 and n2=14. 
2- Data from the first group was further subdivided into a development set (75%), and a held-out test set
3- The development data set was used for five-fold cross-validated model hyperparameter tuning based on the weighted F1 metric
4- The best performing model hyperparameters were used to assess how well the model performs on unseen test data from the same group of individuals on whose data the model was trained
5- The same model was applied to held out test data from the second group of patient datasets, in order to assess how well the model performs on a completely different group of individuals, whose data were not included in the training datasets
6- Due to the strong imbalance of heart beat abnormality class probabilities, stratified sampling was applied at each data splitting stage in order to approximately preserve the relative frequencies of normal vs abnormal heart beats, as well as the four individual heart beat abnormality types.
7- When available on sklearn classifier object API's, the option class_weights='balanced' was used when specifying models, in order to take class imbalance into account when computing cost functions

Four models were considered:

1- Baseline model -  Random guess from a multinomial distribution based on class probabilities for each of the five possible classes
Five-fold cross-validation, each time estimating class prior probabilities from training data, and generating predictions for held out data. The multinomial random guess baseline model performed on average at 59% weighted F1 accuracy.

2- k Nearest Neighbors (kNN) - Hyperparameter = Neighborhood size (optimal k=3). Best performing model. 
Weighted average F1 = 99% when training and testing on data from same group; = 80% when applied to a novel group of individuals
(However, unweighted F1 drops down to 48% in the latter case, suggesting that the shapes of abnormal heart beat ECGs are less consistent across individuals than normal ECG patterns)

3- Logistic regression (LogReg) - Hyperparameter = regularization strength (optimal C=100, light regularization). Lowest performing  
Weighted average F1 = 90% when training and testing on data from same group; = 64% when applied to a novel group of individuals
(unweighted F1 drops down to 31% in the latter case, indicating poor transfer of learned abnormality parameters across groups)

3- Balanced random forest - Hyperparameters (optimized with grid search): number of trees   (optimal = 1000, maximum considered); max number of features at each split (optimal = log2(number of ECG time points)). Second best performing model.
Weighted average F1 = 98% when training and testing on data from same group; = 78% when applied to a novel group of individuals
(unweighted F1 drops down to 43% in the latter case, similar to the other models)


# Conclusions and Future Directions

Highly accurate classification of abnormalities in ECG-recorded single heart beats (importantly, with relatively low quality single channel ECG data, as might be expected in IoT devices such as smart watches) is possible using commonly employed machine learning techniques - provided that extensive labeled training data is available from the same group of individuals whose ECG will later be monitored for heart abnormalities. 

However, there is considerably room for improvement when training models with one group of individual, and applying the same models to an entirely different group of individuals. Labeling 2.5K single heart beats manually in order to generate expert-labeled training data is very expensive and highly impractical for any large-scale ECG monitoring application. Better transfer of learning is an important open question

For the immediate future:

Use single-channel continuous electrocardiogram (ECG) time-series recordings to train a deep learning network in order to:
1) Detect individual heartbeats; quantify onset, peak latency, duration
2) Detect irregular heartbeats; classify anomaly type
3) Flag time windows during which hear beats are arhythmic (i.e., irregular R-R intervals)
4) Identify patients whose heartbeats patterns are anomalous (e.g., aim to diagnose heart arrhythmia or atrilial fibrillation)
5) Apply ECG anomaly detection algorithm trained on one clinical dataset to ECG datasets from other sources (e.g., transfer learning)
6) Most importantly for developing a real-world clinical application, explore appropriate semi-supervised and transfer learning approaches to make best use of available expert-labeled training data




