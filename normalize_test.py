import csv
import json
import numpy as np
import pandas as pd
import open3d as o3d
from datetime import datetime
from pathlib import Path
import bee_utils as bee

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

pc = np.loadtxt("D:/Bibliothek/Workspace/_Studium/MA/datasets/insect/100ms_2048pts_farthest_point-ds_sor-nr_2024-07-03_20-32-37/bee/bee_h1_0_13.csv",
                 delimiter=',', skiprows=1, usecols=(0,1,2)).astype(np.float32)
print(pc[:3,:])

npc = pc_normalize(pc)
print(npc[:3,:])
print(np.mean(npc, axis=0))
print(np.min(npc, axis=0))
print(np.max(npc, axis=0))



