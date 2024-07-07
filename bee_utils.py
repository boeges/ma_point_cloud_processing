# Purpose: Defines common util functions and classes for several scripts.

import re
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

SCENE_IDS = list(SCENE_ID_ALIASES.keys())
SCENE_NAMES = [s[0] for s in SCENE_ID_ALIASES.values()]

# zb "1_l-l-l_trajectories_2024-05-29_15-27-12"
DIR_NAME_PATTERN = re.compile(r"^(.+)_trajectories.*")

def get_date_time_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# return short name (first 3 characters)
def full_class_to_short_class(cla:str, default=None):
    return CLASS_ABBREVIATIONS.get(cla.lower(), default)

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

def scene_id_to_short_id(scene_id:str):
    return SCENE_ID_ALIASES[scene_id][1]
    
def dir_to_scene_name(trajectory_dir_name:str):
    matches = re.findall(DIR_NAME_PATTERN, trajectory_dir_name)
    if len(matches) == 0:
        raise RuntimeError(trajectory_dir_name, " doesnt match pattern!")
    scene_name = matches[0]
    return scene_name








# DEBUG
if __name__ == "__main__":
    print("FULL_CLASS_NAMES", CLASSES)
    print("SCENE_IDS", SCENE_IDS)
    print("SCENE_NAMES", SCENE_NAMES)