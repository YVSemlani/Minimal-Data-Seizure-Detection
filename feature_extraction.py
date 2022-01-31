import numpy as np

#bandpass filters

from scipy.signal import butter, sosfilt, sosfreqz

fs = 256 # sample rate

def butter_bandpass(lowcut, highcut, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, order=5):
    sos = butter_bandpass(lowcut, highcut, order=order)
    y = sosfilt(sos, data)
    return y

def M_bands(data, M, rangepoints = (0, 512), minfreq = 0.5, maxfreq = 25):
    split = (maxfreq - minfreq) / M
    bands = []
    for i in range(M):
        lowcut = minfreq + i*split
        highcut = minfreq + (i+1)*split
        filtered_data = butter_bandpass_filter(data, lowcut, highcut)
        bands.append(filtered_data)
    return np.array(bands)

# powervalues

from scipy.signal import welch
from scipy.integrate import simps

def powervals(data, low=0.5, high=25):
    fs = 256 # sampling frequency
    freqs, psd = welch(data, fs, nperseg=2*fs)
    idx_delta = np.logical_and(freqs >= low, freqs <= high)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
    # Compute the absolute power by approximating the area under the curve
    delta_power = simps(psd[idx_delta], dx=freq_res)
    return delta_power