# 

import csv
from datetime import datetime
from pathlib import Path

# in csv cols: x_float, y:float, t:int, p:0/1
IN_DIR = Path("../../datasets/catch_and_release/3d_shifted/segmented")
OUT_DIR = IN_DIR / "prepared_renamed"

T_SCALE = 1.0

CLASS_MAP = {
    "apis":"bee",
    "bombhum":"bum",
    "bombpasc":"bum",
}

# original files and ids
FIDS = {
    "slave_apis_10_19_1000.csv":0,
    "slave_apis_10_19_1500.csv":1,
    "slave_apis_10_19_200.csv":2,
    "slave_apis_10_19_2_1000.csv":3,
    "slave_apis_10_19_2_1500.csv":4,
    "slave_apis_10_19_2_200.csv":5,
    "slave_apis_10_19_2_500.csv":6,
    "slave_apis_10_19_4_100.csv":7,
    "slave_apis_10_19_4_1500.csv":8,
    "slave_apis_10_19_4_500.csv":9,
    "slave_apis_10_19_500.csv":10,
    "slave_apis_10_19_5_1000.csv":11,
    "slave_apis_10_19_5_1500.csv":12,
    "slave_apis_10_19_5_200.csv":13,
    "slave_apis_10_19_5_500.csv":14,
    "slave_apis_10_30_3_1000.csv":15,
    "slave_apis_10_30_3_1500.csv":16,
    "slave_apis_10_30_3_200.csv":17,
    "slave_apis_10_30_3_500.csv":18,
    "slave_bombhum_10_49_1_100.csv":19,
    "slave_bombhum_10_49_1_1000.csv":20,
    "slave_bombhum_10_49_1_200.csv":21,
    "slave_bombhum_10_49_1_500.csv":22,
    "slave_bombhum_11_06_1_100.csv":23,
    "slave_bombhum_11_06_1_1000.csv":24,
    "slave_bombhum_11_06_1_500.csv":25,
    "slave_bombhum_11_30_100.csv":26,
    "slave_bombhum_11_30_1000.csv":27,
    "slave_bombhum_11_30_500.csv":28,
    "slave_bombhum_11_34_100.csv":29,
    "slave_bombhum_11_34_1500.csv":30,
    "slave_bombhum_11_34_500.csv":31,
    "slave_bombpasc_11_29_100.csv":32,
    "slave_bombpasc_11_29_1000.csv":33,
    "slave_bombpasc_11_29_1500.csv":34,
    "slave_bombpasc_11_29_200.csv":35,
    "slave_bombpasc_11_29_500.csv":36,
    "slave_bombpasc_11_33_100.csv":37,
    "slave_bombpasc_11_33_1000.csv":38,
    "slave_bombpasc_11_33_1500.csv":39,
    "slave_bombpasc_11_33_200.csv":40,
    "slave_bombpasc_11_34_1000.csv":41,
    "slave_bombpasc_11_34_1500.csv":42,
    "slave_bombpasc_11_34_200.csv":43,
    "slave_bombpasc_11_34_500.csv":44,
}


############################ MAIN ##################################
if __name__ == "__main__":
    # dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # in_fps = [file for file in Path("../../datasets/catch_and_release/3d_shifted").iterdir() if file.is_file() and file.name.endswith("00.csv")]
    # for i,f in enumerate(in_fps):
    #     a = f.name
    #     print(f"\"{a}\":{i},")
    # exit()

    in_fps = [file for file in IN_DIR.iterdir() if file.is_file() and file.name.endswith("00 - Cloud.txt")] # txt or csv!!

    for in_fp in in_fps:
        # in: slave_apis_10_19_2_200 - Cloud.txt
        fn = in_fp.name
        fn1 = fn.replace(" - Cloud.txt",".csv")
        fnarr = fn.split("_")
        orig_clas = fnarr[1]
        new_clas = CLASS_MAP[orig_clas]
        fnum = FIDS[fn1]
        

        min_x=9999
        min_y=9999
        min_t=9999999
        count = 0
        with open(in_fp, 'r') as in_f:
            reader = csv.reader(in_f)
            for rrow in reader:
                count += 1
                x = int(float(rrow[0]))
                y = int(float(rrow[1]))
                t = float(rrow[2])
                if x < min_x: min_x=x
                if y < min_y: min_y=y
                if t < min_t: min_t=t


        # out: 1_bee_pts1234_start0.csv
        new_fn = f"{fnum}_{new_clas}_pts{count}_start0.csv"
        print(fn, new_fn)
        out_fp = OUT_DIR / new_fn

        with open(in_fp, 'r') as in_f, open(out_fp, 'w', newline='') as out_f:
            reader = csv.reader(in_f)
            writer = csv.writer(out_f)

            writer.writerow(["x","y","t"])

            for rrow in reader:
                x = int(float(rrow[0])) - min_x
                y = int(float(rrow[1])) - min_y
                t = (float(rrow[2]) - min_t) / T_SCALE
                # p = rrow[3]
                writer.writerow([x,y,t])

