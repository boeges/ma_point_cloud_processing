



from pathlib import Path
import bee_utils as bee
import random
import math
import pandas as pd
import numpy as np
from datetime import datetime



def create_split(dataset_dir:Path, train_size, max_train_percent=0.5):
    """
    creates a train/test split file as txt.

    Args:
        dataset_dir (Path): root to dataset
        train_size (_type_): if float: in percent; if int: absolute size
        max_train_percent (float, optional): only used if train_size is absolute; Max percentage to take of a class for train split. Defaults to 0.5.
    """

    if isinstance(train_size, int) or train_size > 1.0:
        # number of samples in train split is absolute! (K-Shot)
        train_size = int(train_size)
        split_str = f"{train_size}shot"
        is_percentage = False
    else:
        is_percentage = True
        split_str = f"{int(train_size*100):0>2}{int(100-train_size*100):0>2}"

    print("Using", "percentual" if is_percentage else "absolute", "split!")
    print("split string:", split_str)

    # Find all csv files in CSV_DIR
    files = list(dataset_dir.glob("*/*.csv"))

    print("found", len(files), "files")

    classes = list(bee.CLASS_ABBREVIATIONS.keys())

    cm = {k:[] for k in classes}


    for f in files:
        fn = f.name
        clas = f.parent.name
        # print(clas, fn)
        cm[clas].append(f)

    # remove empty
    cm = {k:v for k,v in cm.items() if len(v) > 0}

    for k,v in cm.items():
        print(k, len(v))

    ltrain = []
    ltest = []

    for k,l in cm.items():
        random.shuffle(l)
        if is_percentage:
            ntrain = int(math.ceil(len(l) * train_size))
        else:
            ntrain_max = int(math.ceil(len(l) * max_train_percent))
            ntrain = train_size
            if ntrain > ntrain_max:
                print("WARNING: ntrain is greater than max train percentage for class", k, ". Only taking", ntrain_max, "samples!")
                ntrain = ntrain_max
        taken_train = l[:ntrain]
        taken_test = l[ntrain:]
        print(f"{k:>16}, files: {len(l):>4}, train size:{len(taken_train):>4}, test size {len(taken_test):>4}")

        ltrain += taken_train
        ltest += taken_test

    print("combined size:", len(files), " train size:", len(ltrain), " test size:", len(ltest), " train ratio:", len(ltrain)/(len(ltrain)+len(ltest)))

    # csv format: [train|test], class, file id
    fo = Path(dataset_dir / f"train_test_split_{split_str}.txt")
    with open(fo, "w") as file:
        for f in ltrain:
            t = "train"
            fn = f.name
            fid = "_".join(fn.replace(".csv","").split("_")[-3:])
            clas = f.parent.name
            # print(f"  {t},{clas},{fid}")
            file.write(f"{t},{clas},{fid}\n")
        for f in ltest:
            t = "test"
            fn = f.name
            fid = "_".join(fn.replace(".csv","").split("_")[-3:])
            clas = f.parent.name
            # print(f"  {t},{clas},{fid}")
            file.write(f"{t},{clas},{fid}\n")



def create_split2(dataset_dir:Path, train_size, max_train_percent=0.5):
    """
    creates a train/test split file as txt.

    Args:
        dataset_dir (Path): root to dataset
        train_size (_type_): if float: in percent; if int: absolute size
        max_train_percent (float, optional): only used if train_size is absolute; Max percentage to take of a class for train split. Defaults to 0.5.
    """

    if isinstance(train_size, int) or train_size > 1.0:
        # number of samples in train split is absolute! (K-Shot)
        train_size = int(train_size)
        split_str = f"{train_size}shot"
        is_percentage = False
    else:
        is_percentage = True
        split_str = f"{int(train_size*100):0>2}{int(100-train_size*100):0>2}"

    print("Using", "percentual" if is_percentage else "absolute", "split!")
    print("split string:", split_str)

    # Find all csv files in CSV_DIR
    files = list(dataset_dir.glob("*/*.csv"))

    print("found", len(files), "files")

    l = []
    for f in files:
        fn = f.name
        clas = f.parent.name
        pth = clas+"/"+fn
        scene_id, traj_index, frag_id = bee.frag_filename_to_id(fn)
        traj_id = scene_id+"_"+str(traj_index)
        l.append([scene_id, traj_index, traj_id, frag_id, clas, pth])

    df = pd.DataFrame(l, columns=["scene", "traj_index", "traj_id", "frag", "class", "path"])
    print(df.head())

    train_indices = []
    test_indices = []

    for clas in df["class"].unique():
        class_df = df[df["class"] == clas]

        # Get groups and their sizes
        group_sizes = class_df.groupby("traj_id").size().reset_index(name='size')

        # Calculate the target number of samples for the train set
        total_samples = group_sizes['size'].sum()
        if is_percentage:
            ntrain = int(math.ceil(total_samples * train_size))
        else:
            ntrain_max = int(math.ceil(total_samples * max_train_percent))
            ntrain = train_size
            if ntrain > ntrain_max:
                print("WARNING: ntrain is greater than max train percentage for class", clas, ". Only taking", ntrain_max, "samples!")
                ntrain = ntrain_max

        ll = []
        for _ in range(100):
            # Shuffle groups
            group_sizes1 = group_sizes.sample(frac=1)

            minus = random.choice([0,1,2,3,4,5])
            target = ntrain-minus
            train_inds, test_inds = find_split(class_df, group_sizes1, target)
            dist = abs(ntrain-len(train_inds))
            ll.append([train_inds, test_inds, dist])

        # sort by dist
        ll = sorted(ll, key=lambda v: v[2])
        train_inds = ll[0][0]
        test_inds = ll[0][1]
        print(clas, " total samples:", total_samples, " target:", ntrain, total_samples-ntrain, " split:", len(train_inds), len(test_inds))

        # add to global list
        train_indices.extend(train_inds)
        test_indices.extend(test_inds)

    # Split the dataframe into train and test sets based on indices
    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]

    print(train_df.head())
    print(test_df.head())
    print(df.index.__len__(), train_df.index.__len__(), test_df.index.__len__())

    # csv format: [train|test], class, file id
    fo = Path(dataset_dir / f"train_test_split_{split_str}.txt")
    if fo.exists():
        dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fo = Path(dataset_dir / f"train_test_split_{split_str}_{dt}.txt")
        print(fo)

    with open(fo, "w") as file:
        for _,r in train_df.iterrows():
            t = "train"
            fid = "_".join([r["scene"], str(r["traj_index"]), str(r["frag"])])
            clas = r["class"]
            # print(f"  {t},{clas},{fid}")
            file.write(f"{t},{clas},{fid}\n")
        for _,r in test_df.iterrows():
            t = "test"
            fid = "_".join([r["scene"], str(r["traj_index"]), str(r["frag"])])
            clas = r["class"]
            # print(f"  {t},{clas},{fid}")
            file.write(f"{t},{clas},{fid}\n")


def find_split(samples, frags_per_instance, ntrain):
    for n in range(1,len(frags_per_instance.index)):
        # Take first n fragments from instance
        selection = frags_per_instance.iloc[:n,:]
        frag_count = selection['size'].sum()
        if frag_count >= ntrain:
            # Take all fragments from these instances
            train = samples[samples["traj_id"].isin(selection["traj_id"])].index
            # Take all fragments that are NOT from these instances
            test = samples[~samples["traj_id"].isin(selection["traj_id"])].index
            return train, test
    return [],[]


if __name__ == "__main__":
    DIR = Path("../../datasets/insect/100ms_2048pts_1024minpts_fps-ds_none-nr_norm_shufflet_5")

    # creates a random split over all fragments
    # create_split(DIR, 40, 0.5)

    # creates a split by keeping all fragments of a trajectory in the same split
    create_split2(DIR, 0.7, 0.5)
    # create_split2(DIR, 40, 0.5)
    create_split2(DIR, 100, 0.5)

    