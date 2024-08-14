# Schreibt starttimestamps der einzelnen frames in eine csv Datei.

import csv
import h5py
from pathlib import Path
from datetime import datetime

def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


############################ MAIN ##################################
if __name__ == "__main__":
    H5_DIR = Path("../../datasets/muenster_dataset/wacv2024_ictrap_dataset")
    FRAMETIMES_CSV_DIR = Path("output/mu_h5_frametimes_to_csv")
    FRAMETIMES_CSV_DIR.mkdir(parents=True, exist_ok=True)

    # Find all csv files in CSV_DIR
    h5_filepaths = [file for file in H5_DIR.iterdir() if (file.is_file() and file.name.endswith(".h5"))]
    print(f"Found {len(h5_filepaths)} h5 files")

    for h5_filepath in h5_filepaths:
        filestem = h5_filepath.name.replace(".h5", "")
        frametimes_csv_filepath = FRAMETIMES_CSV_DIR / f"{filestem}_frametimes.csv"

        with h5py.File(h5_filepath, "r") as h5_file, open(frametimes_csv_filepath, 'w', newline='') as frametimes_csv_file:
            triggers = h5_file["EXTERNAL_TRIGGERS/corrected_positive"] # (p, t, channel_id, frame_index)

            # write timestamps of frames
            writer = csv.writer(frametimes_csv_file)
            writer.writerow(("frame_index", "timestamp"))

            for frame_index,trigger in enumerate(triggers):
                trigger_t = int(trigger[1])
                # trigger_frame_index = int(trigger[3]) # dont use this as it starts at -1
                writer.writerow((frame_index, trigger_t))

        print(f"Created {frametimes_csv_filepath}!")
