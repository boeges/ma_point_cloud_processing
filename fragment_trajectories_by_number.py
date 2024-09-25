# Purpose: Fragment individual flight trajectories into smaller parts,
# so that they will fit into PointNet which requires 4096 points.
# Each part is exactly 4096 points. The time length will vary.

import csv
import numpy as np
from datetime import datetime
from pathlib import Path
import pandas as pd

def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


############################ MAIN ##################################
if __name__ == "__main__":
    DATETIME_STR = getDateTimeStr()

    SAVE_FRAGMENTS = False
    SAVE_STATS = True

    MAX_EVENTS_PER_TRAJECTORY = 4096
    # Concurrent/Overlapping windows.
    # 1 means no overlap. When one window is full, it is saved and the net starts.
    # 2 means 2 overlapping windows. When one is 50% full, the next window starts to fill. (amlost) every event will be in two windows.
    CONCURRENT_WINDOWS = 1
    OVERLAP_PHASE_SHIFT = MAX_EVENTS_PER_TRAJECTORY // CONCURRENT_WINDOWS
    SAVE_PARTIAL_TRAJECTORIES = False
    MIN_FILE_SIZE_BYTES = 100 * 1024 # 100KB
    MIN_FILE_SIZE_BYTES = 200

    T_SCALE = 0.002

    TRAJECTORIES_BASE_DIR = Path("output/extracted_trajectories/3_classified")

    LOG_DIR = Path("output/logs/fragmentation_by_num/")

    # scene name starts with
    EXCLUDE_SCENES_FROM_STATS = [
        "_with_bboxes",
        "hn-depth",
    ]

    fragments_stats = []

    # Find all trajectory dirs
    trajectory_dirs = [d for d in TRAJECTORIES_BASE_DIR.glob("*")]

    # Exclude dirs from EXCLUDE_SCENES_FROM_STATS
    trajectory_dirs2 = []
    for trajectory_dir in trajectory_dirs:
        exclude_scene_from_stats = False
        for sn in EXCLUDE_SCENES_FROM_STATS:
            if trajectory_dir.name.startswith(sn):
                exclude_scene_from_stats = True
                break
        if exclude_scene_from_stats:
            print("Excluding scene", trajectory_dir.name)
            continue
        trajectory_dirs2.append(trajectory_dir)
    trajectory_dirs = trajectory_dirs2


    for trajectory_dir in trajectory_dirs:
        scene_id = trajectory_dir.name
        output_base_dir = trajectory_dir / f"fragments_pts{MAX_EVENTS_PER_TRAJECTORY}_cw{CONCURRENT_WINDOWS}"

        print(f"Processing dir: {trajectory_dir}")
        
        # Find all csv files
        csv_paths = [trajectory_dir/file.name for file in trajectory_dir.iterdir() if (file.is_file() and file.name.endswith(".csv") and file.stat().st_size >= MIN_FILE_SIZE_BYTES)]
        print(f"Found {len(csv_paths)} files with >= {MIN_FILE_SIZE_BYTES//1024}KB")

        # TESTING: only use this file!
        # csv_paths = [csv_dir/"0.csv"]
        
        for csv_path in csv_paths:
            print("Fragmenting trajectory file", csv_path, "...")
            print(f"Using MAX_EVENTS_PER_TRAJECTORY={MAX_EVENTS_PER_TRAJECTORY}, CONCURRENT_WINDOWS={CONCURRENT_WINDOWS}")

            # [ [(x,y,t,is_confident), ...], ... ]
            closed_trajectories = []
            open_trajectories = [[] for _ in range(CONCURRENT_WINDOWS)]
            
            # index of the window in the list, that will be closed next because its full
            next_closing_index = 0

            with open(csv_path, 'r') as csv_file:
                reader = csv.reader(csv_file)
                # skip header
                next(reader)

                for event_index, row in enumerate(reader):
                    if event_index % OVERLAP_PHASE_SHIFT == 0:
                        # a trajectory is full
                        # find which
                        next_closing_index = (next_closing_index + 1) % CONCURRENT_WINDOWS
                        trajectory_index = next_closing_index
                        trajectory = open_trajectories[trajectory_index]

                        # print("event_index", event_index, \
                        #     ", trajectory index", trajectory_index, \
                        #     ", size", len(trajectory))

                        # save trj
                        closed_trajectories.append(trajectory)

                        # create new list at the position
                        open_trajectories[trajectory_index] = []

                    # DEBUG
                    # if event_index > 20000:
                    #     break

                    # Append event to all open trj
                    for trajectory in open_trajectories:
                        trajectory.append( (int(row[0]), int(row[1]), float(row[2]), int(row[3])) )

                # save trjs that arent full yet
                for trajectory in open_trajectories:
                    closed_trajectories.append(trajectory)

                print("Fragmented trajectory into", len(closed_trajectories), "parts! Saving ...")

                # Create dir
                csv_stem = csv_path.name.replace(".csv", "")
                csv_stem_arr = csv_stem.split("_")
                traj_id = csv_stem_arr[0]
                clas = csv_stem_arr[1]
                output_dir = output_base_dir / f"{csv_stem}"
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save fragmented trajectories as CSVs
                if SAVE_FRAGMENTS:
                    saved_count, partial_count = 0, 0
                    for i, trajectory in enumerate(closed_trajectories):
                        if len(trajectory) == 0:
                            continue

                        if len(trajectory) < MAX_EVENTS_PER_TRAJECTORY:
                            if not SAVE_PARTIAL_TRAJECTORIES:
                                # skip if option is False
                                continue
                            # trj is not full
                            is_part_suffix = "_partial"
                            partial_count += 1
                        else:
                            is_part_suffix = ""

                        output_path = output_dir / f"frag_{i}{is_part_suffix}.csv"

                        with open(output_path, 'w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(["x", "y", "t", "is_confident"])
                            for tr in trajectory:
                                # limit to n digits after decimal separator
                                writer.writerow((tr[0], tr[1], "%.3f"%tr[2], tr[3]))
                            saved_count += 1

                    print(f"Saved {saved_count} trajectory files ({saved_count-partial_count} complete; {partial_count} partial) in {output_base_dir}")

                if SAVE_STATS:
                    for i, trajectory in enumerate(closed_trajectories):
                        if not SAVE_PARTIAL_TRAJECTORIES and len(trajectory) < MAX_EVENTS_PER_TRAJECTORY:
                            continue
                        
                        start_ts = float(trajectory[0][2]) / T_SCALE
                        end_ts = float(trajectory[-1][2]) / T_SCALE
                        duration = end_ts-start_ts
                        # scene id, traj id, frag id, class, num points, duration in us
                        fragments_stats.append([scene_id, traj_id, i, clas, len(trajectory), duration])


    if SAVE_STATS:
        fragments_df = pd.DataFrame(fragments_stats, \
                columns=["scene", "instance_id", "fragment_id", "class", "frag_event_count", "frag_duration"])
        pth = LOG_DIR / f"fragments_{DATETIME_STR}.csv"
        pth.parent.mkdir(parents=True, exist_ok=True)
        fragments_df.to_csv(pth, index=False, header=True, decimal='.', sep=',', float_format='%.3f')
        print("Saved stats to", pth)

    print("Finished!")





    







                













