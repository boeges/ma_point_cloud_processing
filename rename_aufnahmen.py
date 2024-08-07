
""" 
Purpose: Rename files captured with the prophesee dvs sensor.
Use predefined mappings.

Example: fe_recording2024-07-18T11-42-08.mp4  ->  mb-bee1-1.mp4

"""


from pathlib import Path
import os



DIR = Path("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-08-06_libellen/exported_csv")
TARGET_DIR = DIR #/ "renamed"

# Find all csv files in CSV_DIR
files = [file for file in DIR.iterdir() if file.is_file() and file.name.startswith("fe_recording")]

# for f in files:
#     print(f.name)


# map old to new filename scheme
# from: (to, scene_name)
range_map = {
    "2024-07-18T14-34-27": ("2024-07-18T14-56-28", "dra1"),
    "2024-07-18T15-53-35": ("2024-07-18T16-22-11", "dra2"),
}


# erst alle aufnahmen pro scene sammeln in map (zb "bee1")
captures_by_scene = {}
range_key = None
found_range_end = False

c = 0
for f in files:
    fn = f.name
    fn = fn[12:31] # extract only datetime

    if range_key is not None:
        range_end = range_map[range_key][0]
        if found_range_end and fn != range_end:
            # next file after found the range end
            range_key = None
            found_range_end = False

    if range_key is not None:
        range_end = range_map[range_key][0]
        scene = range_map[range_key][1]

        # get list of captures
        captures = captures_by_scene.setdefault(scene, [])
        # add file to scene captures
        # print("Adding", f.name, "to", scene, "as range")
        captures.append(f)
        c+=1

        if range_end == fn:
            # this is the end of the current range
            # there could be more files with the end name
            found_range_end = True
            # range_key = None

    elif fn in range_map.keys():
        range_end = range_map[fn][0]
        scene = range_map[fn][1]

        # get list of captures
        captures = captures_by_scene.setdefault(scene, [])
        # add file to scene captures
        # print("Adding", f.name, "to", scene, "as single or range start")
        captures.append(f)
        c+=1

        if range_end is not None and range_end != fn:
            # new open range mapping
            range_key = fn
            found_range_end = False
        
print(c, len(files))

# dann sortieren und scenennamen zuordnen ("bee1-1", "bee1-2", ...)

# cap id is the datetime
cap_ids_by_scene =  {k:[] for k in captures_by_scene.keys()}

for k,captures in captures_by_scene.items():
    # print(k)
    for cap in captures:
        fn = cap.name
        fn = fn[12:31] # extract only datetime
        if fn not in cap_ids_by_scene[k]:
            cap_ids_by_scene[k].append(fn)

cap_ids_by_scene =  {k:sorted(v) for k,v in cap_ids_by_scene.items()}

mappings = {}

for k,cids in cap_ids_by_scene.items():
    for i,cid in enumerate(cids):
        new_id = f"mb-{k}-{i+1}"
        mappings[cid] = new_id

for o,n in mappings.items():
    print(o,n)

for f in files:
    fn = f.name
    fn = fn[12:31] # extract only datetime

    new_id = mappings[fn]
    if new_id is not None:
        new_filename = f.name.replace("fe_recording"+fn, new_id)
        new_f = TARGET_DIR / new_filename
        print(f.name, " -> ", new_filename)
        os.rename(str(f), str(new_f))




