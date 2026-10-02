import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import gaussian_filter

# magic iypython, non-blocking mode for figures
%matplotlib qt

EEGFILENAME   = "2026-10-01-eyes_open_eyes_closed"
ONSETFILENAME = "onset_timestamp_alpha.txt"
SFREQ       = 256
FIRSTTASK   = 5  # first marker FIRSTTASK seconds after time stamp
TASKDUR     = 30 # 30 seconds per task
NTASKS      = 6  # 

# def für labels
def get_labels(dframe):
    ch_names = list(df.columns)
    cropped_labels = [item.split("_")[1] for item in ch_names]
    return(cropped_labels)

# read data
df = pd.read_csv("data/" + EEGFILENAME + "_eeg.csv        ")

## read time stamps
# 5 s bsl;  3 * (10s open, 10s closed)
with open(EEGFILENAME + '.txt', 'r', encoding='utf-8') as file:
    first_line = file.readline()
    first_value = first_line.strip().split('\t')[0]
dt1 = datetime.fromisoformat(first_value.strip())

# onset time 
with open(ONSETFILENAME, "r", encoding="utf-8") as datei:
    exp_start = datei.read()
dt2 = datetime.fromisoformat(exp_start)

tdelta = dt2 - dt1
tdiff  = tdelta.total_seconds() + 5 # plus 5 because first event is 5s after time stamp
tdiff_idx = (df.time - tdiff).abs().idxmin()

# remove time and generate labels
df = df.drop(columns=["time"])
labels = get_labels(df)

## add marker column
df["MARKER"] = 0
marker_indices = tdiff_idx + np.arange(0, TASKDUR * NTASKS + 1, TASKDUR) * 256
df.loc[marker_indices, "MARKER"] = 1
labels.append("MARKER") 

# MNE expects shape (n_channels, n_times) and volts but data is in microvolts
# transpose and multiply
data = df.to_numpy().T * 1e-6  
types = ['eeg'] * 4 + ['stim']
info = mne.create_info(ch_names=labels, sfreq = SFREQ, ch_types=types)
raw = mne.io.RawArray(data, info)
raw.set_montage("standard_1020", on_missing="warn")

# remove onset and offset, filter >1Hz
raw_cropped = raw.copy().crop(tmin=2.0, tmax = raw.times[-1]-2.0)
raw_cropped.filter(l_freq=1.0, h_freq=None, fir_design="firwin")
raw_cropped.apply_function(lambda x: (x - x.mean())) # de-mean

# ICA
ica = mne.preprocessing.ICA(n_components=4, random_state=97, max_iter="auto")
ica.fit(raw_cropped)
ica.plot_properties(raw_cropped, picks=[0, 1, 2,3])
ica.exclude = [0] # here 0
raw_cleaned = raw_cropped.copy()
ica.apply(raw_cleaned)
raw_cleaned.plot(scalings=dict(eeg=50e-6))

## tfr and plot
freqs = np.arange(1, 31, 1)          # 1-30 Hz
n_cycles = freqs * 5.0               # window length scales with frequency
tfr = raw_cleaned.compute_tfr("morlet", freqs=freqs, n_cycles=n_cycles)
# average over channels, log-scaled relative to the mean at start (eyes open)
tfr.plot(picks = ['TP9', 'TP10'],
    combine="mean",
     baseline=(None, 5),
     mode="percent",
     vlim = (-6,6))

# plot spectrum
psd = raw_cleaned.compute_psd(
    method="welch",
    fmin=1, fmax=20,
    n_fft=1024,        # FFT length in samples (4 s at 256 Hz, 0.25 Hz resolution)
    n_per_seg=512,     
    n_overlap=256,     # overlap in samples 
    window="hann",
)
psd.plot()

# plot Hilbert Amplitude
from scipy.fft import next_fast_len
raw_alpha = raw_cleaned.copy().filter(l_freq=8.0, h_freq=11.0, fir_design="firwin")
n_fft     = next_fast_len(raw_alpha.n_times)
raw_env   = raw_alpha.copy().apply_hilbert(envelope=True, n_fft=n_fft)

times = raw_env.times
env   = raw_env.get_data() * 1e6          # back to µV, shape (n_channels, n_times)
alpha = raw_alpha.get_data() * 1e6

# One channel: filtered signal with its envelope
#raw_cropped.plot(scalings=dict(eeg=50e-6))
avghilbert      = np.mean(env[[0,3]], axis=0) # avg of TP9 und TP10
smoothedhilbert =  gaussian_filter(avghilbert, sigma=256.0*3)

fig, (ax_hilb, ax_marker) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), 
                                        gridspec_kw={'height_ratios': [3, 1]})

ax_hilb.plot(times, smoothedhilbert, lw=1, label="Hilbert amplitude 8 - 11 Hz (smoothed 3s)")
ax_hilb.set_xlabel("Time (s)")
ax_hilb.set_ylabel("Amplitude (µV)")
ax_hilb.set_title("TP9, TP10")#raw_env.ch_names[ch])
ax_hilb.legend()

marker_channel = 'MARKER' 
raw_marker = raw_cleaned.copy().pick(marker_channel)
marker_data, marker_times = raw_marker[:, :]

ax_marker.plot(marker_times, marker_data[0]*1e+6, color='crimson', linewidth=1.5, label='Marker Signal')
ax_marker.set_ylabel('Marker')
ax_marker.set_xlabel('Time (s)')
ax_marker.grid(True, linestyle='--', alpha=0.6)

eyes_opened_position = marker_indices[[0,2,4]] / SFREQ + 15
eyes_closed_position = marker_indices[[1,3,5]] / SFREQ + 15

for x in eyes_opened_position:
    ax_marker.text(x, 0.5, 'auf', 
                   color='black', fontsize=11, 
                   horizontalalignment='center', verticalalignment='center')

for x in eyes_closed_position:
    ax_marker.text(x, 0.5, 'zu', 
                   color='black', fontsize=11, 
                   horizontalalignment='center', verticalalignment='center')

plt.tight_layout()
plt.show()



