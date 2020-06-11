
def load_ecg_data(filename='shelve_ECG_segments_mitbih.out'):
    '''
    :param filename:
    :return: data_all, annotations_all, onsets_all, pt_codes_all
    '''

    import shelve

    shelf = shelve.open(filename,'r') # 'n' for new

    vars = list(shelf.keys())
    # ['data_all', 'annotations_all', 'onsets_all', 'pt_codes_all' ]

    shelf = shelve.open(filename)
    for key in shelf:
        globals()[key]=shelf[key]
    shelf.close()