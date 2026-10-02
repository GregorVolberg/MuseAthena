# https://github.com/DominiqueMakowski/OpenMuse
# https://brainflow.org/2026-05-08-muse-anthena/

# install openmuse (here dedicated conda environment "openmuse")
# conda activate openmuse
OpenMuse find
#Searching for Muses (max. 10 seconds)...
# Found device MuseS-218B, MAC Address 00:55:DA:BA:21:8B
OpenMuse record --address 00:55:DA:BA:21:8B --preset p1035 --outfile 2026-09-22-Muse_sleep.txt
OpenMuse stream --address 00:55:DA:BA:21:8B --preset p1035 --record 2026-09-22-Muse_sleep_stream_long.txt

import pandas as pd
import OpenMuse
import matplotlib.pyplot as plt

with open("eyes_open_eyes_closed.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()
data = OpenMuse.decode_rawdata(messages)

# Plot Movement Data
data["EEG"].plot(
    x="time",
    y=["EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10"],
    subplots=True
)
plt.show()

# Reihenfolge stimmt nicht; korrekt ist {time, TP10, AF8, AF7, TP9}


# now for Brainflow

import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter

params = BrainFlowInputParams()
params.timeout = 15
preset = BrainFlowPresets.DEFAULT_PRESET
Id = BoardIds.MUSE_S_ATHENA_BOARD.value
params.other_info = 'preset=p20;low_latency=true' # preset to p20?
board = BoardShim(Id, params)

try:
    board.prepare_session()
    board.start_stream()
    try:
        time.sleep(10)
        eeg_data = board.get_board_data(preset=preset)
        imu_data = board.get_board_data(preset=BrainFlowPresets.AUXILIARY_PRESET)
        optical_data = board.get_board_data(preset=BrainFlowPresets.ANCILLARY_PRESET)
    finally:
        board.stop_stream()
finally:
    if board.is_prepared():
        board.release_session()

DataFilter.write_file(eeg_data, 'data/muse_anthena_eeg_bf.csv', 'w')
DataFilter.write_file(imu_data, 'data/muse_anthena_accel_gyro_bf.csv', 'w')
DataFilter.write_file(optical_data, 'data/muse_anthena_optics_battery_bf.csv', 'w')

print("EEG-Kanäle (Spalten-Indizes):", BoardShim.get_eeg_channels(Id, preset))
print("Sonstige Kanäle:", BoardShim.get_other_channels(Id, preset))
print("Package-Num-Kanal:", BoardShim.get_package_num_channel(Id, preset))
print("Timestamp-Kanal:", BoardShim.get_timestamp_channel(Id, preset))
print("Marker-Kanal:", BoardShim.get_marker_channel(Id, preset))

import matplotlib.pyplot as plt
import numpy as np

ts = eeg_data[10,:]
dfs = np.diff(ts)


# read muse data
import numpy as np
import pandas as pd
import OpenMuse
import os
import matplotlib.pyplot as plt

def zscore (dframe):
    rtrn = (dframe - dframe.mean()) / dframe.std()
    return(rtrn)

eegdata = "2026-09-22-Muse_sleep_stream.txt"
with open(eegdata, "r", encoding="utf-8") as f:
    messages = f.readlines()
data = OpenMuse.decode_rawdata(messages)

# returns dict(n=5) of pandas data frames:
# dict_keys(['EEG', 'ACCGYRO', 'OPTICS', 'BATTERY', 'Unknown'])
eeg = data["EEG"].loc[:,"time":"EEG_TP10"]

labels = pd.Index.to_list(eeg.columns)[1:5]
#cropped_labels = [item.split("_")[1] for item in labels]
fig, axs = plt.subplots(nrows = len(labels), ncols = 1, 
                        sharex=True, sharey=True)

for ax, ch in zip(axs, labels):
    ax.plot(eeg["time"], zscore(eeg[ch]), linewidth=0.8)
    ax.set_ylabel(ch)
    ax.grid(True, alpha=0.3)

fig.suptitle('Continuous EEG Muse Athena')
fig.supylabel('Amplitude (z)')
plt.show()

plt.plot(, zscore(eeg.loc[:,"EEG_TP9":"EEG_TP10"]), label = labels[1:5])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (z)")
plt.show()

eeg.plot.line(index = "time")


# Plot Movement Data
data["ACCGYRO"].plot(
    x="time",
    y=["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"],
    subplots=True
)
plt.show()