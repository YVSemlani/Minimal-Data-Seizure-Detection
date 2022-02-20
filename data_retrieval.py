import numpy as np
from tqdm import tqdm

import mne

from feature_extraction import M_bands, powervals

mne.set_log_level(40)

def seizure_timestamps(summaries, seizurelines):      
    index = 0
    timestamps = {}
    for line in summaries:
        if any(item == line[11:] for item in seizurelines) and line[11:] != '':
            crawler = 0
            file = []
            while summaries[index+crawler] != '':
                file.append(summaries[index+crawler])
                crawler += 1
            filetimes = []
            startmark = False
            endmark = False
            for entry in file:
                if 'Seizure Start Time' in entry:
                    start = ''.join(i for i in entry if i.isdigit())
                    startmark = True
                if 'Seizure End Time' in entry:
                    end = ''.join(i for i in entry if i.isdigit())
                    endmark = True
                if 'Seizure' in entry and any(char.isdigit for char in entry[:17]):
                    if 'Start Time' in entry[10:]:
                        start = ''.join(i for i in entry[10:] if i.isdigit())
                        startmark = True
                    elif 'End Time' in entry[10:]:
                        end = ''.join(i for i in entry[10:] if i.isdigit())
                        endmark = True
                if endmark and startmark:
                    filetimes.append((start, end))
                    startmark = False
                    endmark = False
            timestamps[line[11:]] = filetimes
        index += 1
    return timestamps
def classify(seizureclass, timestamps):
    # seizureclass: array with desired classification dimensions
    # timestamps: dict of files: [timestamp(s)] staggered by patient
    sf = 256
    epochlength = sf * 200
    index = 0
    for file in timestamps.values():
        for time in file:
            delta = int(time[1]) - int(time[0]) // 2
            if delta >= epochlength // sf:
                delta = epochlength // sf
            seizureclass[index][:delta] = int(1)
            index += 1
    return seizureclass

def seizurevector_retrieval(timestamps, initialseizure=0, endseizure=141, channel1=None, channel2=None):
    sf = 256
    s = sf * 100
    seizures = []
    index = 0
    indices = []
    for file in tqdm(list(timestamps.keys())[initialseizure:endseizure]):
        # iterates through each record with seizures (should be 142)
        try:
            path = f'../chbmit/1.0.0/{file[:5]}/{file}'
            # print(path)
            data = mne.io.read_raw_edf(path, preload=True)
            raw_data = data.get_data()
            if channel1 != None and channel2 != None:
                raw_data = raw_data[channel1:channel2]
            for seizure in timestamps[file]:
                # iterates through each seizure within a record; total seizures should be 197
                start = sf * int(seizure[0])
                seizurevector = raw_data.take(indices=range(start-s, start+s), axis=1)
                #print(f'Length of Seizure Vector {((start + s) - (start - s)) / 256}')
                bands = M_bands(seizurevector, 8)
                bands = np.array(np.split(bands, 512, 2))
                bands = np.swapaxes(bands, 0, 3)
                bands = np.swapaxes(bands, 0, 2)
                bands = np.swapaxes(bands, 0, 1)
                pv = np.apply_along_axis(powervals, 3, bands)
                pv = np.swapaxes(pv, 0, 2)
                pv = np.swapaxes(pv, 0, 1)
                pv = np.hstack((pv[channel1:channel2]))
                seizures.append(pv)
            index += 1
        except Exception as e:
            indices.append(index)
            print(index)
            index += 1
            print(e)
    try:
        seizures = np.array(seizures)
        return seizures, indices
    except Exception as e:
        print(e)
        return seizures, indices