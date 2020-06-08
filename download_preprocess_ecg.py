


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import medfilt
#%%
#!wget  https://www.kaggle.com/mondejar/mitbih-database/download

def download_mitbih_ecg_data():
    import requests

    print('Downloaded MIT-BIH ECG data from Physionet')

    url = 'https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip'
    # alternative    'https://www.kaggle.com/mondejar/mitbih-database/download'

    r = requests.get(url, allow_redirects=True)
    print(r)
    open('mit_bih_ecg_database', 'wb').write(r.content)

    print('Downloaded MIT-BIH ECG data')

    from zipfile import ZipFile

    with ZipFile('mit_bih_ecg_database.zip', 'r') as zipObj:
       zipObj.extractall('mit_bih_csv')

#%%

# Read ECG datasets for each patient into memory
# Segment continous ECG into short epochs, each corresponding to a single heart beat
# Downsample, smooth, detrend and baseline correct each time segment
# Combine ECG time segments into a single matrix (rows = heart beats, columns = time points
# Save data matrix to disk (in present directory)
#
# dir_ecg_csv = 'mit_bih_csv' # ecg data directory
#
# # Patient codes for train and test data sets
# pts_train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
# pts_test = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
#
# PTs = sorted(pts_train + pts_test)

def read_patient_ecg(patient_id):

    '''
    For each patient, read continuous ECG time series and event (herat beat) annotations
    Extract ECG segments for each heart beat
    Apply linear drift correction (detrend), baseline correction, minimal smoothing

    :param patient_id:
    :return: data matrix,
    '''

    from scipy.signal import detrend

    fn_csv = ''.join([dir_ecg_csv, '/', str(patient_id), '.csv'])
    fn_onset = ''.join([dir_ecg_csv, '/', str(patient_id), 'annotations.txt'])

    df = pd.read_csv( fn_csv ).iloc[:,1]
    annotation = pd.read_csv( fn_onset,
                         header=None, skiprows=2, sep=r"[ ]{2,}", #"r"\s*",
                         engine='python' )
    onset = annotation.iloc[:,1]
    annot = annotation.iloc[:,2]

    # exclude first and last pulse
    annot = annot[2:-2]
    onset = onset[2:-2]

    # extract and preprocess ecg segments
    bef, aft = 100, 200 #take 100 samples before, 200 after the ecg peak
    dat = np.zeros((len(onset),bef+aft)) # alloc memory for all ecg segments (rows)
    df = df.rolling(10).mean().rolling(5).median() #smooth ecg time series

    for i, anc in enumerate(onset):
        ts = detrend(df[anc-bef:anc+aft]) # remove drift
        ts -= ts[:bef].mean()  # adjust pre-peak baseline
        ts /= ts.max() / 100 # scale each segment to max amp = 100
        dat[i,:] = ts.reshape(1,-1)

    pt_code = (np.ones((len(onset),1)) * patient_id).astype(int)

    # downsample data matrix - return every third sample
    return dat[:,::3], onset.values.reshape(-1,1), annot.values.reshape(-1,1), pt_code

#%%

# Loop over patient datasets
# load and preprocess patient ecg data
# keep segmented ECG data, onsets of ecg segments relative to continuous recording,
# abnormality labels (annotations), and corresponding patient code for each segment in arrays

def preprocess_ecg(PTs=None):
    '''
    :param PTs: list of patient codes
    :return: data matrix, event onsets, anomaly labels, patient code per heart beat
    '''
    data= np.array([]) # single data matrix for all ecg segments from all patients
    onsets = np.array([]) # heart beat onsets, relative to continuous ECG
    annotations = np.array([]) # anomaly labels for individual heart beats
    pt_codes = np.array([]) # for each heart beat (row in data matrix), record patient code

    for i, patient_id in enumerate(PTs):
        print(i)
        dat, onset, annot, pt_code = read_patient_ecg(patient_id)
        if i == 0:
            data, onsets, annotations, pt_codes = dat, onset, annot, pt_code
        else:
            data = np.concatenate([data,dat], axis=0)
            onsets = np.concatenate([onsets,onset], axis=0)
            annotations = np.concatenate([annotations, annot], axis=0)
            pt_codes = np.concatenate([pt_codes, pt_code], axis=0)
        print(patient_id)
        return data, annotations, onsets, pt_codes

#%%

# save preprocessed ECG data to disk using shelve

import shelve

#data_all, annotations_all, onsets_all, pt_codes_all = data, annotations, onsets, pt_codes


def save_ecg_data(filename='shelve_ECG_segments_mitbih.out')
    shelf = shelve.open(filename,'n') # 'n' for new

    for key in ['data_all', 'annotations_all', 'onsets_all', 'pt_codes_all' ]:
        try:
            shelf[key] = globals()[key]
        except TypeError:
            print('ERROR shelving: {0}'.format(key))
    shelf.close()
    pass


#%%



if __name__ == '__main__':

    download_mitbih_ecg_data()  # download raw ECG datasets (MIT-BIH Arrythmia Database, PhysioNet)

    dir_ecg_csv = 'mit_bih_csv'  # ecg data directory

    # Patient codes for train and test data sets
    pts_train = \
        [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
    pts_test = \
        [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

    PTs = sorted(pts_train + pts_test)


    data_all, annotations_all, onsets_all, pt_codes_all = preprocess_ecg(PTs)

    filename = 'shelve_ECG_segments_mitbih.out'
    save_ecg_data(filename)
    print('saved preprocessed ecg data in : ' + filename )

    # To read ecg data into memory
