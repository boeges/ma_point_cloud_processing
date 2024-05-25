# Wandelt gesamte Punktwolkenmenge in CSV um. (Nicht einzelne Bahnen).

import cv2
import csv
import h5py
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

width = 1280
height = 720

filename = "1_l-l-l"
# filename = "3_m-h-h"
events_filepath = f"../../datasets/muenster_dataset/wacv2024_ictrap_dataset/{filename}.h5"
csv_filepath = f"{filename}_{getDateTimeStr()}.csv"

t_factor = 0.001

data = [
    ['x', 'y', 't', "is_insect"]
]


# event zB (133, 716, 1, 1475064)
with h5py.File(events_filepath, "r") as f:
    events = f["CD/events"]
    first_timestamp = events[0][3]

    for event in events:
        if first_timestamp+1000*1000*2 < event[3]:
            # stop after x sec
            break

        data.append([event[0], event[1], int(float(event[3])*t_factor)])


# Write CSV
with open(csv_filepath, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Created {csv_filepath}!")
