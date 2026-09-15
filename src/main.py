# https://github.com/DominiqueMakowski/OpenMuse
# https://brainflow.org/2026-05-08-muse-anthena/

import mne

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

params = BrainFlowInputParams()
board = BoardShim(BoardIds.MUSE_S_ANTHENA_BOARD.value, params)