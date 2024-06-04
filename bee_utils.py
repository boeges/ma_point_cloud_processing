# Purpose: Defines common util functions and classes for several scripts.

from datetime import datetime

def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def full_class_to_short_class(cla:str, default=None):
    mapp = {
        "bee": "bee", 
        "butterfly": "but", 
        "dragonfly": "dra", 
        "wasp": "was"
    }
    return mapp.get(cla.lower(), default)

def parse_full_class_name(cla:str, default=None):
    if cla=="b" or cla=="bee":
        return "bee"
    elif cla=="u" or cla=="f" or cla=="but" or cla=="butterfly":
        return "butterfly"
    elif cla=="d" or cla=="dra" or cla=="dragonfly":
       return "dragonfly"
    elif cla=="w" or cla=="was" or cla=="wasp":
        return "wasp"
    # Also for case "i" or "insect"
    return default





