import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import gaussian_filter

# magic iypython, non-blocking mode for figures
%matplotlib qt

EEGFILENAME   = "2026-10-01-compute_or_relax"
ONSETFILENAME = "onset_timestamp_theta.txt"
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
df = pd.read_csv("data/" + EEGFILENAME + "_eeg.csv")

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

# spectrum
psd = raw_cleaned.compute_psd(
    method="welch",
    fmin=1, fmax=35,
    n_fft=1024,        # FFT length in samples (4 s at 256 Hz, 0.25 Hz resolution)
    n_per_seg=512,     
    n_overlap=256,     # overlap in samples 
    window="hann",
)

psds, freqs = psd.get_data(return_freqs = True)

fig = psd.plot(show=False)
ax = fig.axes[0]
ax.annotate('Theta', 
            xy=(6, 2.0),       # Pfeil Ende
            xytext=(7, 5.5),     # Pfeil anfang
            color='darkblue', fontsize=11,
            arrowprops=dict(
                facecolor='black',   # Pfeilfarbe
                shrink=0.08,         # Abstand zu Text/Zielpunkt
                width=1,           # Dicke des Pfeilschafts
                headwidth=5          # Breite der Pfeilspitze
            ))
plt.show()


# epoching for theta / beta ratio
# need to re-scale marker channel to integer
stim       = raw_cleaned.get_data(picks="MARKER")[0]
evt_marker = (stim > 0).astype(int)
idx = raw_cleaned.ch_names.index("MARKER")
raw_cleaned._data[idx] = evt_marker
raw_cleaned.set_channel_types({"MARKER": "stim"})

events = mne.find_events(raw_cleaned, stim_channel="MARKER", shortest_event=1)
labels_per_event = ["compute", "relax"]*3 + ["end"]
event_id         = {"compute": 1, "relax": 2, "end": 3}  # your own mapping

assert len(labels_per_event) == len(events), "marker count != label count"
events[:, 2] = [event_id[l] for l in labels_per_event]
epochs = mne.Epochs(raw_cleaned, events, event_id=event_id, tmin=0, tmax=30,
                    baseline=(None, 30), picks="eeg", preload=True)

psd1 = epochs["compute"].compute_psd(
    method="welch",
    fmin=1, fmax=35,
    n_fft=1024,        # FFT length in samples (4 s at 256 Hz, 0.25 Hz resolution)
    n_per_seg=512,     
    n_overlap=256,     # overlap in samples 
    window="hann",
)

psd2 = epochs["relax"].compute_psd(
    method="welch",
    fmin=1, fmax=35,
    n_fft=1024,        # FFT length in samples (4 s at 256 Hz, 0.25 Hz resolution)
    n_per_seg=512,     
    n_overlap=256,     # overlap in samples 
    window="hann",
)

compute_psd, freqs = psd1.get_data(return_freqs=True)
relax_psd          = psd2.get_data()

AF7 = labels.index("AF7")

i4  = np.searchsorted(freqs, 4, side="left")    # erster Index mit freqs >= 4
i7  = np.searchsorted(freqs, 7, side="right")   # erster Index mit freqs > 7
i13 = np.searchsorted(freqs, 13, side="left")   # erster Index mit freqs >= 13
i30 = np.searchsorted(freqs, 30, side="right")  # erster Index mit freqs >= 13

theta = compute_psd[:, AF7, i4:i7].mean(axis=1)
beta  = compute_psd[:, AF7, i13:i30].mean(axis=1)
tbratio_compute = theta/beta

theta = relax_psd[:, AF7, i4:i7].mean(axis=1)
beta  = relax_psd[:, AF7, i13:i30].mean(axis=1)
tbratio_relax = theta/beta

tbratio_compute
tbratio_relax


x = np.arange(len(tbratio_compute))   # 0, 1, 2
w = 0.35                              # Balkenbreite

fig, ax = plt.subplots()
ax.bar(x - w/2, tbratio_compute, w, label="compute")
ax.bar(x + w/2, tbratio_relax, w, label="relax")

ax.set_xticks(x)
ax.set_xticklabels([f"Trial {i+1}" for i in x])
ax.set_ylabel("Theta/Beta-Ratio")
ax.legend()
plt.show()



