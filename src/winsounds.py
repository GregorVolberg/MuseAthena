import winsound
import time
from datetime import datetime, timezone

def wait_and_beep(seconds_to_wait: int, hzbeep: int):
    time.sleep(seconds_to_wait)
    winsound.Beep(hzbeep, 500)


utc_now = datetime.now(timezone.utc)
print(f"[BEEP] UTC-Zeit: " + utc_now.isoformat())
with open("onset_timestamp_theta.txt", "w", encoding = "utf-8") as datei:
    datei.write(utc_now.isoformat())
wait_and_beep(5, 1000)  # Start: Augen auf bzw rechnen (17-er Schritte von dreistelliger zahl > 500)
wait_and_beep(30, 500) # Augen zu bzw entspannen
wait_and_beep(30, 1000) # Augen auf
wait_and_beep(30, 500) # Augen zu
wait_and_beep(30, 1000) # Augen auf
wait_and_beep(30, 500) # Augen zu
wait_and_beep(30, 1000) # Ende
wait_and_beep(0.2, 1000) # Ende
