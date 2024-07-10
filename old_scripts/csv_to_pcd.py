# In den Labels von DarkLabel werden IDs manchmal mehrfach vergeben. Um die Flugbahnen zu trennen wird dieses Skript benutzt.

import csv
from pathlib import Path
from datetime import datetime

def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def write_pcd_part(part_rows, pcd_filepath:Path):
    pcd_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(pcd_filepath, 'w') as pcd_file:
        pcd_file.write("# .PCD v.7 - Point Cloud Data file format\n")
        pcd_file.write("VERSION .7\n")
        pcd_file.write("FIELDS x y z\n")
        pcd_file.write("SIZE 4 4 4\n")
        pcd_file.write("TYPE U U F\n")
        pcd_file.write(f"WIDTH {len(part_rows)}\n")
        pcd_file.write("HEIGHT 1\n")
        pcd_file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        pcd_file.write(f"WIDTH {len(part_rows)}\n")
        pcd_file.write("DATA ascii\n")
        for part_row in part_rows:
            pcd_file.write(f"{part_row[0]} {part_row[1]} {part_row[2]:.4f}\n")


############################ MAIN ##################################
if __name__ == "__main__":
    DATETIME_STR = getDateTimeStr()

    T_FACTOR = 0.001
    CSV_HAS_HEADER_ROW = False
    MAX_ROWS_PER_PART = 500_000 # for output file
    PRINT_PROGRESS_EVERY_N_PERCENT = 1

    CSV_DIR = Path("../../datasets/Insektenklassifikation")
    # CSV_DIR = Path("../../datasets/juli_instance_segmented")
    OUTPUT_DIR = Path("output") / "csv_to_pcl" / f"{DATETIME_STR}_{MAX_ROWS_PER_PART}pts"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    csv_filenames = [
        # "1_l-l-l",
        # "2_l-h-l",
        # "3_m-h-h",
        # "4_m-m-h",
        # "5_h-l-h",
        # "6_h-h-h_filtered",
        "hauptsächlichBienen1.csv",
        # "vieleSchmetterlinge2.csv"
    ]

    for csv_filename in csv_filenames:
        csv_filestem = csv_filename.replace(".csv", "")
        csv_filepath = CSV_DIR / csv_filename

        print(f"Processing file: {csv_filepath}")

        # Find total row count for tracking progress
        row_count = sum(1 for _ in open(csv_filepath))
        last_progress_percent = 0
        part_index = 1
        part_rows = []

        with open(csv_filepath, 'r') as csv_file:
            reader = csv.reader(csv_file)

            # skip header (if there is no header this will just skip the first event)
            next(reader)

            for row_index, row in enumerate(reader):
                # row: (x, y, t, p)
                timestamp = float(row[2])
                scaled_timestamp = timestamp * T_FACTOR
                part_rows.append( (int(row[0]), int(row[1]), scaled_timestamp) )
                if len(part_rows) >= MAX_ROWS_PER_PART:
                    # save part
                    part_dir = OUTPUT_DIR / csv_filestem
                    pcd_filepath = part_dir / f"part{part_index:0>3}.pcd"
                    write_pcd_part(part_rows, pcd_filepath)
                    part_rows.clear()
                    part_index += 1

            if len(part_rows) >= 4096:
                # save last part
                part_dir = OUTPUT_DIR / csv_filestem
                pcd_filepath = part_dir / f"part{part_index:0>3}.pcd"
                write_pcd_part(part_rows, pcd_filepath)
                part_rows.clear()
                part_index += 1



