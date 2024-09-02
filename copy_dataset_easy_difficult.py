
import os
import shutil
import pandas as pd
import bee_utils as bee
from pathlib import Path


EASY_DIFF_PATH = Path("output/fragment_labels/fragment_easy_difficult_ds5_1.csv")
DATASET_DIR = Path("../../datasets/insect/100ms_4096pts_rnd-ds_sor-nr_norm_shufflet_5")
OUTPUT_DIR = Path("../../datasets/insect/100ms_4096pts_rnd-ds_sor-nr_norm_shufflet_5_diff")

# Find all csv files in CSV_DIR
files = list(DATASET_DIR.glob("*/*.csv"))

# create target directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for d in DATASET_DIR.glob("*"):
    if "." not in d.name:
        # is directory
        (OUTPUT_DIR / d.name).mkdir(parents=False, exist_ok=True)

print("input dataset samples found:", len(files))

df = pd.read_csv(EASY_DIFF_PATH, header="infer")
print(df)
df["is_difficult"] = df["is_difficult"].apply(lambda v: v>=1.0)
print(df["is_difficult"].unique())

sids = df["sample_id"].to_list()
isdiffs = df["is_difficult"].to_list()
difficult_map = dict(zip(sids,isdiffs))

count = 0
count_to_insect = 0
for f in files:
    fn = f.name
    sample_id = fn.replace(".csv","")
    # print(sample_id)
    is_diff = difficult_map.get(sample_id, False)

    if is_diff and f.parent.name != "insect":
        # if sample is difficult move it to "insect" dir; If it is not already in "insect"
        f2 = OUTPUT_DIR / "insect" / f.name
        shutil.copyfile(f, f2)
        count_to_insect+=1
    else:
        # if sample is not difficult: copy to same class dir
        f2 = OUTPUT_DIR / f.parent.name / f.name
        shutil.copyfile(f, f2)
    count+=1



print("files copied:", count, ". Moved to insect:", count_to_insect)
    
    

