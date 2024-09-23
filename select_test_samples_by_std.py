
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import bee_utils as bee

# columns ["split", "sample_id"]
SPLIT_PATH = Path("../../datasets/insect/100ms_2048pts_1024minpts_fps-ds_none-nr_norm_shufflet_5/train_test_split_7030.txt")

# columns ['scene', 'instance_id', 'fragment_id', 'class', 'traj_evnt_count', 'traj_len_s', 'frag_evnt_count', 'frag_len_s', 'frag_std']
STATS_PATH = Path("output/statistics/hist/tf100ms_ds5/all_fragments.csv")

MIN_STD = 4


def to_sample_id(x):
    return x["scene"] + "_" + str(x["instance_id"]) + "_" + str(x["fragment_id"])


if __name__ == "__main__":
    split_df = pd.read_csv(SPLIT_PATH, names=["split","class","sample_id"])
    stats_df = pd.read_csv(STATS_PATH)

    # "aaa_bbb_ccc"
    stats_df["sample_id"] = stats_df.apply(to_sample_id, axis=1)
    stats_df = stats_df[["sample_id","frag_std"]]

    df = pd.merge(split_df, stats_df, left_on="sample_id", right_on="sample_id", how="left", validate="one_to_one")

    print(df)

    df = df[df.frag_std > MIN_STD]

    print(df)
    print(df["frag_std"].min())

    

    out_split_path = SPLIT_PATH.parent / (SPLIT_PATH.stem + f"_minstd{MIN_STD}.txt")
    df[["split","class","sample_id"]].to_csv(out_split_path, index=False, header=False, decimal='.', sep=',', float_format='%.3f')
    print("Saved", out_split_path)