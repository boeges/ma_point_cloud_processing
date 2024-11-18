# Wandelt die gesamte Punktwolkenmenge aus der H5 in eine CSV um. (Nicht einzelne Bahnen).

import csv
import h5py
from pathlib import Path
from datetime import datetime

def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


############################ MAIN ##################################
if __name__ == "__main__":
    WIDTH = 1280
    HEIGHT = 720

    # Must be an integer! e.g. 1 or 10
    PRINT_PROGRESS_EVERY_N_PERCENT = 1

    H5_DIR = Path("../../datasets/muenster_dataset/wacv2024_ictrap_dataset")
    CSV_OUTPUT_DIR = Path("output/mu_h5_to_csv")

    # Find all csv files in CSV_DIR
    h5_filepaths = [file for file in H5_DIR.iterdir() if (file.is_file() and file.name.endswith("3_m-h-h.h5"))] #  ONLY mu-3 !!!!!!!!!!!!!
    print(f"Found {len(h5_filepaths)} h5 files")

    for h5_filepath in h5_filepaths:
        filestem = h5_filepath.name.replace(".h5", "")
        csv_filepath = CSV_OUTPUT_DIR / f"{filestem}.csv"

        t_factor = 1

        # event zB (133, 716, 1, 1475064)
        with h5py.File(h5_filepath, "r") as h5_file, open(csv_filepath, 'w', newline='') as csv_file:
            events = h5_file["CD/events"]
            first_timestamp = events[0][3]

            writer = csv.writer(csv_file)
            writer.writerow(['x', 'y', 't', 'p'])

            event_count = len(events)
            last_progress_percent = 0

            print(f"Processing file: {h5_filepath.name} with {event_count} events")
            print("Progress (%): ", end="")
            for i, event in enumerate(events):
                # Write CSV
                writer.writerow([event[0], event[1], int(float(event[3])*t_factor), event[2]])

                # print progress
                progress_percent = int(i / event_count * 100)
                if progress_percent >= last_progress_percent + PRINT_PROGRESS_EVERY_N_PERCENT:
                    last_progress_percent = progress_percent
                    print(progress_percent, ", ", sep="", end="")

                # if first_timestamp+1000*1000*2 < event[3]:
                #     # stop after x sec
                #     break


        print(f"Created {csv_filepath}!")
