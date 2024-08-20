# 

import csv
from datetime import datetime
from pathlib import Path

# in csv cols: x_float, y:float, t:int, p:0/1
IN_DIR = Path("../../datasets/catch_and_release/3d_shifted_2/segmented")
OUT_DIR = IN_DIR / "prepared_renamed"
OUT_DIR.mkdir(exist_ok=True)

T_SCALE = 1.0

CLASS_MAP = {
    "apis":"bee",
    "bombhum":"bum",
    "bombpasc":"bum",
}

# original files and ids
FIDS = {
    "slave_apis_10_05_1073.csv":0,
    "slave_apis_10_05_173.csv":1,
    "slave_apis_10_05_323.csv":2,
    "slave_apis_10_05_473.csv":3,
    "slave_apis_10_05_623.csv":4,
    "slave_apis_10_05_773.csv":5,
    "slave_apis_10_05_923.csv":6,
    "slave_apis_10_11_2_1093.csv":7,
    "slave_apis_10_11_2_193.csv":8,
    "slave_apis_10_11_2_343.csv":9,
    "slave_apis_10_11_2_493.csv":10,
    "slave_apis_10_11_2_643.csv":11,
    "slave_apis_10_11_2_793.csv":12,
    "slave_apis_10_11_2_943.csv":13,
    "slave_apis_10_19_1016.csv":14,
    "slave_apis_10_19_1166.csv":15,
    "slave_apis_10_19_266.csv":16,
    "slave_apis_10_19_2_1019.csv":17,
    "slave_apis_10_19_2_1169.csv":18,
    "slave_apis_10_19_2_269.csv":19,
    "slave_apis_10_19_2_419.csv":20,
    "slave_apis_10_19_2_569.csv":21,
    "slave_apis_10_19_2_719.csv":22,
    "slave_apis_10_19_2_869.csv":23,
    "slave_apis_10_19_416.csv":24,
    "slave_apis_10_19_4_1142.csv":25,
    "slave_apis_10_19_4_1292.csv":26,
    "slave_apis_10_19_4_1442.csv":27,
    "slave_apis_10_19_4_542.csv":28,
    "slave_apis_10_19_4_692.csv":29,
    "slave_apis_10_19_4_842.csv":30,
    "slave_apis_10_19_4_992.csv":31,
    "slave_apis_10_19_566.csv":32,
    "slave_apis_10_19_5_1020.csv":33,
    "slave_apis_10_19_5_1170.csv":34,
    "slave_apis_10_19_5_270.csv":35,
    "slave_apis_10_19_5_420.csv":36,
    "slave_apis_10_19_5_570.csv":37,
    "slave_apis_10_19_5_720.csv":38,
    "slave_apis_10_19_5_870.csv":39,
    "slave_apis_10_19_716.csv":40,
    "slave_apis_10_19_866.csv":41,
    "slave_apis_10_30_3_1105.csv":42,
    "slave_apis_10_30_3_205.csv":43,
    "slave_apis_10_30_3_355.csv":44,
    "slave_apis_10_30_3_505.csv":45,
    "slave_apis_10_30_3_655.csv":46,
    "slave_apis_10_30_3_805.csv":47,
    "slave_apis_10_30_3_955.csv":48,
    "slave_bombhum_10_49_1_1137.csv":49,
    "slave_bombhum_10_49_1_1287.csv":50,
    "slave_bombhum_10_49_1_387.csv":51,
    "slave_bombhum_10_49_1_537.csv":52,
    "slave_bombhum_10_49_1_687.csv":53,
    "slave_bombhum_10_49_1_837.csv":54,
    "slave_bombhum_10_49_1_987.csv":55,
    "slave_bombhum_11_06_1_1056.csv":56,
    "slave_bombhum_11_06_1_1206.csv":57,
    "slave_bombhum_11_06_1_1356.csv":58,
    "slave_bombhum_11_06_1_1506.csv":59,
    "slave_bombhum_11_06_1_606.csv":60,
    "slave_bombhum_11_06_1_756.csv":61,
    "slave_bombhum_11_06_1_906.csv":62,
    "slave_bombhum_11_30_1064.csv":63,
    "slave_bombhum_11_30_1214.csv":64,
    "slave_bombhum_11_30_1364.csv":65,
    "slave_bombhum_11_30_1514.csv":66,
    "slave_bombhum_11_30_1664.csv":67,
    "slave_bombhum_11_30_1814.csv":68,
    "slave_bombhum_11_30_914.csv":69,
    "slave_bombhum_11_34_1065.csv":70,
    "slave_bombhum_11_34_1215.csv":71,
    "slave_bombhum_11_34_1365.csv":72,
    "slave_bombhum_11_34_1515.csv":73,
    "slave_bombhum_11_34_1665.csv":74,
    "slave_bombhum_11_34_1815.csv":75,
    "slave_bombhum_11_34_915.csv":76,
    "slave_bombpasc_11_29_1009.csv":77,
    "slave_bombpasc_11_29_1159.csv":78,
    "slave_bombpasc_11_29_1309.csv":79,
    "slave_bombpasc_11_29_1459.csv":80,
    "slave_bombpasc_11_29_1609.csv":81,
    "slave_bombpasc_11_29_1759.csv":82,
    "slave_bombpasc_11_29_859.csv":83,
    "slave_bombpasc_11_33_1056.csv":84,
    "slave_bombpasc_11_33_1206.csv":85,
    "slave_bombpasc_11_33_1356.csv":86,
    "slave_bombpasc_11_33_1506.csv":87,
    "slave_bombpasc_11_33_1656.csv":88,
    "slave_bombpasc_11_33_1806.csv":89,
    "slave_bombpasc_11_33_906.csv":90,
    "slave_bombpasc_11_34_1057.csv":91,
    "slave_bombpasc_11_34_1207.csv":92,
    "slave_bombpasc_11_34_1357.csv":93,
    "slave_bombpasc_11_34_1507.csv":94,
    "slave_bombpasc_11_34_1657.csv":95,
    "slave_bombpasc_11_34_757.csv":96,
    "slave_bombpasc_11_34_907.csv":97,
}


############################ MAIN ##################################
if __name__ == "__main__":
    # dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # in_fps = [file for file in Path("../../datasets/catch_and_release/3d_shifted_2").iterdir() if file.is_file() and file.name.endswith(".csv")]
    # for i,f in enumerate(in_fps):
    #     a = f.name
    #     print(f"\"{a}\":{i},")
    # exit()

    in_fps = [file for file in IN_DIR.iterdir() if file.is_file() and file.name.endswith(" - Cloud.txt")] # txt or csv!!

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

            for i,rrow in enumerate(reader):
                x = int(float(rrow[0])) - min_x
                y = int(float(rrow[1])) - min_y
                t = (float(rrow[2]) - min_t) / T_SCALE
                # p = rrow[3]
                if i>2 and t < 0.0001:
                    # corrupt event with row: 0,0,0.0
                    continue

                writer.writerow([x,y,t])

