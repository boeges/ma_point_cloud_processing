
""" 
Statische train/test Auftelung eines Datensatzes erstellen.
Prozentual pro klasse elemente auswählen.

Erzeugt csv datei (.txt), z.B. "train_test_split_2080.txt"
Spalten: split type, class, file id

Beispiel:
train,insect,hn-but-2_5_1
train,insect,hn-but-2_23_1
...
test,bee,hn-bee-1_11_1
test,bee,mu-3_9_259
...

"""


from pathlib import Path
import bee_utils as bee
import random
import math


DIR = Path("../../datasets/insect/100ms_2048pts_fps-ds_sor-nr_1")

TRAIN_SPLIT = 0.2
SPLIT_STR = f"{int(TRAIN_SPLIT*100):0>2}{int(100-TRAIN_SPLIT*100):0>2}"
print("split", SPLIT_STR)

# Find all csv files in CSV_DIR
files = list(DIR.glob("*/*.csv"))

print("files", len(files))

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
    ntrain = int(math.ceil(len(l) * TRAIN_SPLIT))
    taken_train = l[:ntrain]
    taken_test = l[ntrain:]
    print(len(l),len(taken_train),len(taken_test))

    ltrain += taken_train
    ltest += taken_test

print(len(files), len(ltrain), len(ltest), len(ltrain)/(len(ltrain)+len(ltest)))

# csv format: [train|test], class, file id
fo = Path(DIR / f"train_test_split_{SPLIT_STR}.txt")
with open(fo, "w") as file:
    for f in ltrain:
        t = "train"
        fn = f.name
        fid = "_".join(fn.replace(".csv","").split("_")[-3:])
        clas = f.parent.name
        print(f"  {t},{clas},{fid}")
        file.write(f"{t},{clas},{fid}\n")
    for f in ltest:
        t = "test"
        fn = f.name
        fid = "_".join(fn.replace(".csv","").split("_")[-3:])
        clas = f.parent.name
        print(f"  {t},{clas},{fid}")
        file.write(f"{t},{clas},{fid}\n")
