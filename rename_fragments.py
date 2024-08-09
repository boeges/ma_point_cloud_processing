
""" 
Purpose: Rename dataset files.

Example:
..\..\datasets\insect\100ms_2048pts_fps-ds_sor-nr_norm_shufflet_1\wasp\wasp_h9_4_19.csv
to:
..\..\datasets\insect\100ms_2048pts_fps-ds_sor-nr_norm_shufflet_1\wasp\hn-was-2_4_19.csv

"""


from pathlib import Path
import os
import bee_utils as bee


DIR = Path("../../datasets/insect/100ms_4096pts_2048minpts_fps-ds_sor-nr_norm_shufflet_2024-08-09_20-09-37")

# Find all csv files in CSV_DIR
files = list(DIR.glob("*/*.csv"))

print("files", len(files))

classes = list(bee.CLASS_ABBREVIATIONS.keys())

count = 0
for f in files:
    fn = f.name
    for k,v in bee.SCENE_SHORT_ID_ALIASES.items():
        if k in fn:
            sid = v[1]
            fn2 = fn.replace(k, sid)
            for c in classes:
                if c in fn2:
                    fn2 = fn2.replace(c+"_", "")
            f2 = f.parent / fn2
            # print(f, " -> ", f2)
            os.rename(f, f2)
            count+=1
            break

print("moved", count)



