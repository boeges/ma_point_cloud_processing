# 

import csv
from datetime import datetime
from pathlib import Path

# in csv cols: x_float, y:float, t:int, p:0/1
IN_DIR = Path("../../datasets/catch_and_release/3d_shifted")
OUT_DIR = IN_DIR / "prepared"


############################ MAIN ##################################
if __name__ == "__main__":
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    in_fps = [file for file in IN_DIR.iterdir() if file.is_file() and file.name.endswith("00.csv")]

    for in_fp in in_fps:
        # in_fp = IN_DIR / "slave_apis_10_19_2_200.csv"
        out_fp = OUT_DIR / in_fp.name

        with open(in_fp, 'r') as in_f, open(out_fp, 'w', newline='') as out_f:
            reader = csv.reader(in_f)
            writer = csv.writer(out_f)

            for rrow in reader:
                x = int(float(rrow[0]))
                y = int(float(rrow[1]))
                t = rrow[2]
                p = rrow[3]
                writer.writerow([x,y,t,p])

