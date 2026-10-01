# read muse data
import pandas as pd
import OpenMuse
import os

#fname = "2026-09-30-eyes_open_eyes_closed"
fname = "2026-09-30-compute_or_relax"

eegdata = fname + ".txt"
with open(eegdata, "r", encoding="utf-8") as f:
    messages = f.readlines()
data = OpenMuse.decode_rawdata(messages)


# returns dict(n=5) of pandas data frames:
# dict_keys(['EEG', 'ACCGYRO', 'OPTICS', 'BATTERY', 'Unknown'])
eeg     = data["EEG"].loc[:,"time":"EEG_TP10"]
accgyro = data["ACCGYRO"]
optics  = data["OPTICS"]

eeg.to_csv(os.path.join(os.getcwd(), "data", fname + "_eeg.csv"), index=False)
accgyro.to_csv(os.path.join(os.getcwd(), "data", fname + "_accgyro.csv"), index=False)
optics.to_csv(os.path.join(os.getcwd(), "data", fname + "_optics.csv"), index=False)
