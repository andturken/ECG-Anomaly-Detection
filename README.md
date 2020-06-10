# ECG-Anomaly-Detection
Machines Learning for Monitoring Abnormal Heart Beats on Electrocardiogram (ECG) Recordings

# Overview
Use single-channel continuous electrocardiogram (ECG) time-series recordings to train a deep learning network in order to:
1) Detect individual heartbeats; quantify onset, peak latency, duration
2) Detect irregular heartbeats; classify anomaly type
3) Flag time windows during which hearbeats are arhythmic (irregular R-R intervals)
4) Identify patients whose heartbeats patterns are anomalous (e.g., diagnose arhythmia)
5) Apply ECG anomaly detection algorithm trained on one clinical dataset to ECG datasets from other sources (e.g., transfer learning)

# Problem Statement

# Methods

# Data sources:
## MIT-BIH Arrhythmia Database (Requires creating an account to access the data)
https://physionet.org/content/mitdb/1.0.0/
https://console.cloud.google.com/storage/browser/mitdb-1.0.0.physionet.org
Clinical ECG recordings (30 min, dual-lead, 125 Hz) from 48 adults, 23 with rare heart conditions
109,000 manually segmented, expert-labeled heart beat samples
Standard ECG data format (.dat) with expert annotations (.atr, text labels)

WFDB Python libary (https://github.com/MIT-LCP/wfdb-python) is required to read ECG data 

Alternatively,
## Same data, in csv format, are available from Kaggle:    
(requires Kaggle account for access)

## Pre-processed ECG data, in the form of numpy arrays, are available in this repository:
- inset link -


![Heartbeat](ecg_heart_animation.gif)


# Replicating the data analyses steps


The entire library of oral arguments I used can be obtained from the following repository in its ./oyez/cases directory.

The functions I used to parse, combine, and model this data are contained in this repository's src directory.

To replicate the parsing, dataframe creation, and model tuning that this repository covers, one can run the following:

# Clone this repository
git clone https://github.com/andturken/ECG-Anomaly-Detection.git
cd ECG-Anomaly-Detection

#Run Script to create data matrices:

Required Packages: Python 3, scipy, pandas, numpy, matplotlib, sklearn, imbalanced-learn

## Pre-process ECG data
python create_df_and_fit_models_script.py
This will: 1-) Read MIT-BIH ECG data in CSV format into memory 2-) Extract ECG segments corresponding to individual heart beats

The result is saved in ---

## Modeling: 
python 

This script will

## Modeling: Combined unsupervised learning and classification
python

This script will:


# Conclusion


# Future Work






