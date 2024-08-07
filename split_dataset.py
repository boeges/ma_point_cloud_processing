



from pathlib import Path
import bee_utils as bee
import random
import math


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
        print(k, " files:", len(l), " train size:", len(taken_train), " test size", len(taken_test))

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



if __name__ == "__main__":
    DIR = Path("../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_3")

    # TRAIN_SPLIT = 0.7
    # TRAIN_SPLIT = 0.2
    TRAIN_SPLIT = 20

    create_split(DIR, TRAIN_SPLIT, 0.5)

    