# Purpose: Defines common util functions and classes for several scripts.

import re
import pandas as pd
from matplotlib import colors
from datetime import datetime

# First in tuple MUST be 3-character abbreviation!
CLASS_ABBREVIATIONS = {
    "bee":          ("bee","b"),
    "butterfly":    ("but","u","f"),
    "dragonfly":    ("dra","d"),
    "wasp":         ("was","w"),
    "insect":       ("ins","i"),
    "other":        ("oth","o"),  # other objects that are not insects
}

CLASSES = list(CLASS_ABBREVIATIONS.keys())

# scene_id: [full scene name, short scene id]
""" scene_id: [full scene name, short scene id] """
SCENE_ID_ALIASES = {
    # HSNR
    "hn-bee-1":         ["hauptsächlichBienen1", "h1"],
    "hn-bee-2":         ["hauptsächlichBienen2", "h2"],
    "hn-dra-1":         ["libellen1", "h3"],
    "hn-dra-2":         ["libellen2", "h4"],
    "hn-dra-3":         ["libellen3", "h5"],
    "hn-but-1":         ["vieleSchmetterlinge1", "h6"],
    "hn-but-2":         ["vieleSchmetterlinge2", "h7"],
    "hn-was-1":         ["wespen1", "h8"],
    "hn-was-2":         ["wespen2", "h9"],
    "hn-was-3":         ["wespen3", "h10"],
    # Muenster
    "mu-1":             ["1_l-l-l", "m1"],
    "mu-2":             ["2_l-h-l", "m2"],
    "mu-3":             ["3_m-h-h", "m3"],
    "mu-4":             ["4_m-m-h", "m4"],
    "mu-5":             ["5_h-l-h", "m5"],
    "mu-6":             ["6_h-h-h_filtered", "m6"],
}
""" short_id: [scene_name, id] """
SCENE_SHORT_ID_ALIASES = {v[1]: [v[0], k] for k, v in SCENE_ID_ALIASES.items()}
""" scene_name: [short_id, id] """
SCENE_NAME_ALIASES = {v[0]: [v[1], k] for k, v in SCENE_ID_ALIASES.items()}

SCENE_IDS = list(SCENE_ID_ALIASES.keys())
SCENE_NAMES = [s[0] for s in SCENE_ID_ALIASES.values()]

# zb "1_l-l-l_trajectories_2024-05-29_15-27-12"
DIR_NAME_PATTERN = re.compile(r"^(.+)_trajectories.*")

CLASS_COLORS = [
    (1.0, 0.4980392156862745, 0.0, 1.0), # bee, oragne
    (0.30196078431372547, 0.6862745098039216, 0.2901960784313726, 1.0), # butterfly, green
    (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0), # dragonfly, blue
    (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0), # wasp, red
    (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0), # insect, purple
    (1.0, 1.0, 0.2, 1.0), # other, yellow
    (0.6509803921568628, 0.33725490196078434, 0.1568627450980392, 1.0),
    (0.9686274509803922, 0.5058823529411764, 0.7490196078431373, 1.0)
]

CLASS_CMAP = colors.ListedColormap(CLASS_COLORS)


def get_date_time_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def full_class_to_short_class(cla:str, default=None):
    """ return short name (first 3 characters) """
    return CLASS_ABBREVIATIONS.get(cla.lower(), default[:3])[0]

def parse_full_class_name(cla:str, default=None):
    for c,abbr in CLASS_ABBREVIATIONS.items():
        if cla == c or cla in abbr:
            return c
    return default

def scene_name_to_id(scene_name:str):
    for id,aliases in SCENE_ID_ALIASES.items():
        if scene_name == id or scene_name in aliases:
            return id
    raise RuntimeError(f"Scene name {scene_name} not in SCENE_ID_ALIASES!")

def scene_short_id_by_id(scene_id:str):
    return scene_aliases_by_id(scene_id)[1]

def scene_aliases_by_id(scene_id:str):
    """ returns: [scene_name, scene_short_id] """
    return SCENE_ID_ALIASES[scene_id]

def scene_aliases_by_short_id(scene_short_id:str):
    """ returns: [scene_name, scene_id ]"""
    return SCENE_SHORT_ID_ALIASES[scene_short_id]
    
def dir_to_scene_name(trajectory_dir_name:str):
    matches = re.findall(DIR_NAME_PATTERN, trajectory_dir_name)
    if len(matches) == 0:
        raise RuntimeError(trajectory_dir_name, " doesnt match pattern!")
    scene_name = matches[0]
    return scene_name

# make key (scene_id, instance_id, frag_index).
# example: "dragonfly\dragonfly_h3_6_5.csv".
# get "h3_6_5" and split to "h3","6","5".
# (scene_id, instance_id, fragment_index).
# then convert "6" and "5" to int.
def frag_filename_to_id(filename) -> tuple:
    frag_id = filename.replace(".csv","").split("_")[-3:]
    frag_id[1] = int(frag_id[1])
    frag_id[2] = int(frag_id[2])
    frag_id = tuple(frag_id)
    return frag_id

def id_tuple_to_str(id:tuple) -> str:
    return f"{id[0]}_{id[1]}_{id[2]}"

def get_rgba_of_class_index(class_index):
    if isinstance(class_index, pd.Series):
        # convert whole series (pd.Series)
        return class_index.apply(get_rgba_of_class_index)
    # cnvert single int
    return CLASS_COLORS[int(class_index)]
    




# DEBUG
if __name__ == "__main__":
    print("FULL_CLASS_NAMES", CLASSES)
    print("SCENE_IDS", SCENE_IDS)
    print("SCENE_NAMES", SCENE_NAMES)
    print(parse_full_class_name("i", "aaaaa"))

    print({v[1]: [k, v[0]] for k, v in SCENE_ID_ALIASES.items()})