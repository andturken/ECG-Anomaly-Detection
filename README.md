# ECG-Anomaly-Detection
Deep Learning for ECG Anomaly Detection

Use single-channel continuous electrocardiogram (ECG) time-series recordings to train a deep learning network in order to:
1) Detect individual heartbeats; quantify onset, peak latency, duration
2) Detect irregular heartbeats; classify anomaly type
3) Flag time windows during which hearbeats are arhythmic (irregular R-R intervals)
4) Identify patients whose heartbeats patterns are anomalous (e.g., diagnose arhythmia)
5) Apply ECG anomaly detection algorithm trained on one clinical dataset to ECG datasets from other sources (e.g., transfer learning)

Data sources:
MIT-BIH Arrhythmia Database
https://physionet.org/content/mitdb/1.0.0/
https://console.cloud.google.com/storage/browser/mitdb-1.0.0.physionet.org
Clinical ECG recordings (30 min, dual-lead, 125 Hz) from 48 adults, 23 with rare heart conditions
109,000 manually segmented, expert-labeled heart beat samples
Standard ECG data format (.dat) with expert annotations (.atr, text labels)

AHA ECG Dataset
14,000 heart beat samples

