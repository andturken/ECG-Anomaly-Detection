
# Load ECG data matrics to memory

def load_ecg_data_matrix(filename='data_ecg_pickle'):
    '''
    Load ECG data matrix into memory
    Update global variable dict using keys read from pickle file
    '''
    import pickle

    with open('data_ecg.pickle', 'rb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        data_dict = pickle.load(f)

    for key in data_dict.keys():
        globals()[key] = data_dict[key]
        print('loaded ' + key + ' to memory')
        v = globals()[key]
        print( '  ' + str(type(v)) + '       ' + str(v.shape) )

# load_ecg_data_matrix(filename='data_ecg_pickle')
#%%

# List all ECG type annotations, and their frequencies of occurrence
# Will use five most common types
def list_ecg_annotations(annotations_all):
    a = np.unique(annotations_all)
    annotation_count = dict()
    #print('ECG abnormality types and their counts')
    for aa in a:
        annotation_count[aa] = np.sum(annotations_all==aa)
        #L.append((aa, sum(annotations_all==aa)))
        # print(aa, sum(annotations_all==aa))
    return annotation_count

#print(list_ecg_annotations(annotations_all))
#%%
# flag potential ecg artifacts: excessive variability and large negative deflections
# on individual ECG segments

#data_all_shape_before_remove = data_all.shape

def remove_ecg_artifacts(data_all):
    data_all = data_all.copy()
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

    n_removed = sum(np.logical_or(std_th, amp_th))
    n_kept = sum(np.logical_and(~std_th, ~amp_th))
    print( 'Flagged ' + str( n_removed ) + ' out of ' + str( len(std_th)) + ' ECG segments as artifacts ' )
    return data_all

data_all = remove_ecg_artifacts(data_all)