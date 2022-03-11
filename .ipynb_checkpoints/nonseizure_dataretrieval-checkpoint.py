import numpy as np
import pandas as pd
import random

from tqdm import tqdm

import mne

from feature_extraction import M_bands, powervals

mne.set_log_level(40)

def non_seizure_retrieval(recordlines, seizurelines, i1=None, i2=None):
    sf = 256
    s = sf * 100
    seizures = []
    indices = []
    index = 0
    nonseizures = []
    for record in recordlines:
        if record not in seizurelines:
            nonseizures.append(record)
    for file in tqdm(nonseizures):
        try:
            # iterates through each record with seizures (should be 142)
            path = f'../chbmit/1.0.0/{file[:5]}/{file}'
            data = mne.io.read_raw_edf(path, preload=True)
            raw_data = data.get_data()
            if i1 != None and i2 != None:
                raw_data = raw_data[i1:i2]
            duration = len(raw_data[0]) / 256
            start = random.randint(1, duration-1) * sf
            nonseizurevector = raw_data.take(indices=range(start-s, start+s), axis=1)
            # USE FOR DEBUGGING print(f'Length of Seizure Vector {((start + s) - (start - s)) / 256}')
            bands = M_bands(nonseizurevector, 8)
            bands = np.array(np.split(bands, 512, 2))
            bands = np.swapaxes(bands, 0, 3)
            bands = np.swapaxes(bands, 0, 2)
            bands = np.swapaxes(bands, 0, 1)
            pv = np.apply_along_axis(powervals, 3, bands)
            pv = np.swapaxes(pv, 0, 2)
            pv = np.swapaxes(pv, 0, 1)
            pv = np.hstack((pv[0:23]))
            seizures.append(pv)
            index += 1
        except Exception as e:
            indices.append(index)
            print(index)
            index += 1
            print(e)
    try:
        nonseizures = np.array(seizures)
        return nonseizures, indices
    except:
        return nonseizures, indices