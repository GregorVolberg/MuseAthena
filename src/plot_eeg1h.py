# plot eeg 1h
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

fname = Path("data") / "2026-10-08-160300_p20_eeg.csv"
eeg = pd.read_csv(fname)

for col in eeg.columns[1:5]:  # erste 4 Datenspalten nach "time"
    zeeg = (eeg[col] - eeg[col].mean()) / eeg[col].std()
    plt.plot(eeg["time"], zeeg, label=col)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (z)")
plt.legend()
plt.show()
