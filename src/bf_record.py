"""
Kontinuierliches Auslesen eines BrainFlow-Boards (z.B. Muse) und
Schreiben der Daten inkl. Zeitstempel und Kanalnamen in eine CSV-Datei.

Beenden mit Strg+C.
"""

# installiere mit pip setuptools mit niedriger versionsnummer

# pip install "setuptools==70.0.0"
# besser auch niedrige Python-Version, 3.11



import argparse
import csv
import signal
import sys
import time
from datetime import datetime, timezone

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


def build_header(board_id: int) -> list[str]:
    """Erstellt eine Liste von Spaltennamen: Kanalname wo bekannt, sonst 'ch_<i>'."""
    descr = BoardShim.get_board_descr(board_id)
    n_channels = BoardShim.get_num_rows(board_id)

    names = ["" for _ in range(n_channels)]

    # Bekannte Kanaltypen mit Klartext-Namen versehen, sofern vorhanden
    channel_groups = {
        "eeg_channels": "eeg_names",  # eeg hat eigene Namensliste (z.B. TP9, AF7, AF8, TP10)
    }

    if "eeg_channels" in descr and "eeg_names" in descr:
        eeg_names = descr["eeg_names"].split(",")
        for idx, ch in enumerate(descr["eeg_channels"]):
            names[ch] = eeg_names[idx] if idx < len(eeg_names) else f"eeg_{idx}"

    other_groups = [
        "accel_channels", "gyro_channels", "ppg_channels", "eda_channels",
        "eog_channels", "ecg_channels", "emg_channels", "ppg_channels",
        "temperature_channels", "resistance_channels", "other_channels",
        "battery_channel",
    ]
    for group in other_groups:
        if group in descr:
            value = descr[group]
            # battery_channel ist ein einzelner Index, kein List
            if isinstance(value, int):
                names[value] = group.replace("_channel", "")
            else:
                for idx, ch in enumerate(value):
                    label = group.replace("_channels", "")
                    names[ch] = f"{label}_{idx}"

    if "timestamp_channel" in descr:
        names[descr["timestamp_channel"]] = "timestamp"
    if "marker_channel" in descr:
        names[descr["marker_channel"]] = "marker"
    if "package_num_channel" in descr:
        names[descr["package_num_channel"]] = "package_num"

    # Restliche, nicht zugeordnete Kanäle generisch benennen
    for i, n in enumerate(names):
        if not n:
            names[i] = f"ch_{i}"

    return names


def main():
    parser = argparse.ArgumentParser(description="BrainFlow -> CSV Streaming")
    parser.add_argument("--serial-port", type=str, default="",
                         help="z.B. COM5 (Windows) oder /dev/ttyUSB0 (Linux), falls benötigt")
    parser.add_argument("--mac-address", type=str, default="",
                         help="Bluetooth MAC-Adresse, falls benötigt (z.B. für manche Muse-Setups)")
    parser.add_argument("--board-id", type=int, default=BoardIds.MUSE_S_BOARD,
                         help="BrainFlow Board-ID, default: Muse S (siehe BoardIds enum)")
    parser.add_argument("--out", type=str, default=None,
                         help="Ausgabedatei, default: brainflow_data_<timestamp>.csv")
    parser.add_argument("--poll-interval", type=float, default=0.2,
                         help="Polling-Intervall in Sekunden")
    args = parser.parse_args()

    out_path = args.out or f"brainflow_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    params = BrainFlowInputParams()
    if args.serial_port:
        params.serial_port = args.serial_port
    if args.mac_address:
        params.mac_address = args.mac_address

    BoardShim.enable_dev_board_logger()  # bei Bedarf auskommentieren, um Logs zu reduzieren
    board = BoardShim(args.board_id, params)

    header = build_header(args.board_id)
    timestamp_idx = BoardShim.get_timestamp_channel(args.board_id)

    print(f"Verbinde mit Board-ID {args.board_id} ...")
    board.prepare_session()
    board.start_stream()
    print(f"Streaming gestartet. Schreibe nach: {out_path}")
    print("Beenden mit Strg+C.")

    csv_file = open(out_path, "w", newline="")
    writer = csv.writer(csv_file)
    # Zusätzliche, lesbare ISO-Zeitspalte neben dem rohen BrainFlow-Unix-Timestamp
    writer.writerow(header + ["timestamp_iso"])

    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    n_written = 0
    try:
        while running:
            time.sleep(args.poll_interval)
            count = board.get_board_data_count()
            if count == 0:
                continue

            data = board.get_board_data()  # (n_channels, n_samples), leert den Puffer
            n_samples = data.shape[1]

            for i in range(n_samples):
                row = data[:, i].tolist()
                unix_ts = row[timestamp_idx]
                iso_ts = datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()
                writer.writerow(row + [iso_ts])

            n_written += n_samples
            csv_file.flush()  # sicherstellen, dass bei Abbruch nichts verloren geht
            print(f"\r{n_written} Samples geschrieben", end="", flush=True)

    finally:
        print("\nBeende Stream ...")
        # Restliche Daten im Puffer noch sichern
        remaining = board.get_board_data()
        if remaining.shape[1] > 0:
            for i in range(remaining.shape[1]):
                row = remaining[:, i].tolist()
                unix_ts = row[timestamp_idx]
                iso_ts = datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()
                writer.writerow(row + [iso_ts])

        csv_file.close()
        board.stop_stream()
        board.release_session()
        print(f"Fertig. Datei gespeichert unter: {out_path}")


if __name__ == "__main__":
    main()