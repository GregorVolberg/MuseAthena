import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject

def get_labels(dframe):
    ch_names = list(df.columns)
    cropped_labels = [item.split("_")[1] for item in ch_names]
    return(cropped_labels)

csv_path = "data/2026-10-09-233000_p20_eeg.csv"
df = pd.read_csv(csv_path)
df.drop(columns=["time"], inplace=True)

sfreq = 256 
labels = df.columns 

data = df[labels].to_numpy().T 
data = data * 1e-6

cropped_labels = get_labels(df)

# 3. Create a continuous MNE Raw object
info = mne.create_info(ch_names=cropped_labels, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(data, info)

montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

# 4. Filter the data (Crucial for sleep staging: 0.5 - 30 Hz)
raw.filter(l_freq=0.5, h_freq=30.0, fir_design="firwin")

# 5. Generate fixed-length 30-second events
epoch_duration = 30  # seconds
events = mne.make_fixed_length_events(raw, duration=epoch_duration)

# 6. Epoch the data
# tmin=0 and tmax=30 creates non-overlapping 30s windows
epochs = mne.Epochs(
    raw,
    events=events,
    tmin=0.0,
    tmax=epoch_duration,
    baseline=None,
    preload=True,
    reject_by_annotation=False,  # Keep all segments for automated cleaning
)

print(f"Total epochs created: {len(epochs)}")

# 7. Automatically detect and repair/reject artifacts
ar = AutoReject(
    n_interpolate=[1, 2],  # Since you only have 4 channels, interpolate max 1-2
    consensus=[0.2, 0.5, 0.8],
    #ch_types=["eeg"],
    random_state=42,
    n_jobs=-1,
)

epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
print(f"Cleaned epochs remaining: {len(epochs_clean)}")

# 8. Convert clean epochs back to a 3D NumPy array for your ML/DL model
# Shape will be: (n_epochs, n_channels, n_times) -> where n_times = 256 * 30 = 7680
x_data = epochs_clean.get_data()
print(f"Final data shape for sleep staging: {x_data.shape}")
