
import os
import shutil
import pandas as pd
import bee_utils as bee
from pathlib import Path

INSTANCE_CLASS_PATH = Path("output/instance_classes/tsne_inspector/msg_cls5C_e40_bs8_pts4096_split100shot_ds5fps_all_labelled.csv")
DATASET_DIR = Path("../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_5")
OUTPUT_DIR = Path("../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_5_ilp")

# Find all csv files in CSV_DIR
files = list(DATASET_DIR.glob("*/*.csv"))

# create target directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for d in DATASET_DIR.glob("*"):
    if "." not in d.name:
        # is directory
        (OUTPUT_DIR / d.name).mkdir(parents=False, exist_ok=True)

print("input dataset samples found:", len(files))

df = pd.read_csv(INSTANCE_CLASS_PATH, header="infer")
df["scene_instance_id"] = df.apply(lambda v: v["scene_id"]+"_"+str(v["instance_id"]), axis=1)
print(df)


s_i_ids = df["scene_instance_id"].to_list()
classes = df["class"].to_list()
id_class_map = dict(zip(s_i_ids,classes))

print(id_class_map)

copied_count = 0
ignored_count = 0
for f in files:
    fn = f.name
    sample_id = fn.replace(".csv","")
    s_i_id = "_".join(sample_id.split("_")[0:2])
    # print(sample_id)
    new_clas = id_class_map.get(s_i_id, None)

    if new_clas is not None:
        f2 = OUTPUT_DIR / new_clas / f.name
        print("Copying fragment", f.parent.name + "/" + sample_id, "to", new_clas)
        shutil.copyfile(f, f2)
        copied_count += 1
    else:
        print("No class mapping for fragment", f.parent.name + "/" + sample_id)
        ignored_count += 1



print("files copied:", copied_count, ". Ignored:", ignored_count)
    
    

