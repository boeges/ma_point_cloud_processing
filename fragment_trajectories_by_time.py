# Purpose: Fragment individual flight trajectories into smaller parts,
# so that they will fit into PointNet which requires 4096 points.
# Each part has the same time length. 
# Points will be reduced or created to achieve exactly 4096 points.

# TODO move points in (x,y) around origin?

import csv
import json
import numpy as np
import pandas as pd
import open3d as o3d
from datetime import datetime
from pathlib import Path
import bee_utils as bee


class Fragment:
    def __init__(self):
        self.index = None
        self.start = None
        self.original_event_count = None
        self.noise_reduced_event_count = None
        self.sampled_event_count = None
        # original events
        self.events = []
        # processed events as dataframe
        self.events_df = None

def index_or_default(l, value, default):
    try:
        return l.index(value)
    except ValueError:
        return default

# Make and store json of local variables/options
def get_options_json(option_names, vars, json_path=None):
    options = {}
    for name in option_names:
        val = vars[name]
        try:
            if isinstance(val, (str, int, float, list, bool)):
                options[name] = val
            elif isinstance(val, Path):
                options[name] = str(val)
            else:
                options[name] = "unsupported type: " + str(type(val))
        except Exception:
            options[name] = "dangerously unsupported type: " + str(type(val))
    if json_path is not None:
        json.dump(options, open(json_path, "w"), indent=2)
        print("Stored options json to", str(json_path))
    return options

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


############################ MAIN ##################################
if __name__ == "__main__":
    ### OUTPUT options
    DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ADD_TIME_TO_FILENAME = True
    DATETIME_STR_PREFIX = ('_'+DATETIME_STR) if ADD_TIME_TO_FILENAME else ''
    # If the csv contains a EVENTS_CSV_BB_CORNER_COL, 
    # add bbox events to the resulting fragments or remove those events
    ADD_BB_EVENTS = False
    SAVE_PARTIAL_TRAJECTORIES = True
    OUTPUT_DIR_MODE = "dataset_dir" # "parent_dir", "dataset_dir"
    SAVE_STATISTICS = True

    ### DEBUG
    PRINT_PROGRESS_EVERY_N_PERCENT = 1

    ### CSV Format
    # Format for events: (x, y, t (us or ms), p)
    EVENTS_CSV_X_COL = 0
    EVENTS_CSV_Y_COL = 1
    EVENTS_CSV_T_COL = 2
    EVENTS_CSV_P_COL = None
    EVENTS_CSV_IS_CONFIDENT_COL = None
    EVENTS_CSV_FRAME_INDEX_COL = None
    EVENTS_CSV_BB_CORNER_COL = None

    
    ### Fragmentation and other params
    MIN_FILE_SIZE_BYTES = 100 * 1024 # 100KB
    # Precision of the timestamp in fractions per second.
    # For mikroseconds: 1000000, for milliseconds: 1000
    TIMESTEPS_PER_SECOND = 1_000_000
    # The factor with which the timestamps in the trajetory csv have been scaled.
    # If they are unscaled/original use 1.0; If they have been multiplied with 0.002 use 0.002.
    T_SCALE = 0.002
    INV_T_SCALE = 1 / T_SCALE
    # Timely width of a t-bucket in microseconds or milliseconds.
    # 1000 * 100 means 100ms. Thus events in each 100ms interval are collected in a t-bucket.
    # Butterfly has 20 wing beats per second (wbps), bee has 200wbps.
    T_BUCKET_LENGTH = 1000 * 100
    # T_BUCKET_LENGTH = 1000 * 4000 # 4s
    T_BUCKET_LENGTH_MS = int(T_BUCKET_LENGTH / TIMESTEPS_PER_SECOND * 1000)
    # This is our target event count per fragment; Use "all" to keep original event count
    # EVENTS_PER_FRAGMENT = "all"
    EVENTS_PER_FRAGMENT = 4096
    # Min number of events a fragment needs before adding or removing events
    MIN_EVENTS_COUNT = EVENTS_PER_FRAGMENT//2

    DOWNSAMPLE_METHOD = "farthest_point" # "random", "farthest_point"
    DOWNSAMPLE_METHOD_STR = "fps" if "farthest_point" else ("rnd" if "random" else "no")

    NOISE_REDUCTION_METHOD = "sor" # "none", "sor" = statistical outlier removal

    # Normalize point clouds of each fragment
    NORMALIZE = True
    NORMALIZE_STR = "_norm" if NORMALIZE else ""
    # Shuffle points of each fragment
    SHUFFLE_T = True
    SHUFFLE_T_STR = "_shufflet" if SHUFFLE_T else ""

    ### Paths
    # TRAJECTORIES_CSV_DIR = Path("output/extracted_trajectories/2_separated_mu")
    TRAJECTORIES_BASE_DIR = Path("output/extracted_trajectories/3_classified")
    OUTPUT_DATASET_DIR = Path("../../datasets/insect/") / \
        f"{T_BUCKET_LENGTH_MS}ms_{EVENTS_PER_FRAGMENT}pts_{DOWNSAMPLE_METHOD_STR}-ds_{NOISE_REDUCTION_METHOD}-nr"\
        f"{NORMALIZE_STR}{SHUFFLE_T_STR}{DATETIME_STR_PREFIX}"
    LOG_DIR = Path("output/logs/fragmentation") / DATETIME_STR

    # Save parameters to json file
    OPTION_NAMES = [
        "DATETIME_STR",
        "ADD_TIME_TO_FILENAME",
        "ADD_BB_EVENTS",
        "SAVE_PARTIAL_TRAJECTORIES",
        "OUTPUT_DIR_MODE",
        "TRAJECTORIES_BASE_DIR",
        "OUTPUT_DATASET_DIR",
        "MIN_FILE_SIZE_BYTES",
        "TIMESTEPS_PER_SECOND",
        "T_SCALE",
        "T_BUCKET_LENGTH",
        "T_BUCKET_LENGTH_MS",
        "EVENTS_PER_FRAGMENT",
        "MIN_EVENTS_COUNT",
        "DOWNSAMPLE_METHOD",
        "NOISE_REDUCTION_METHOD",
    ]
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    get_options_json(OPTION_NAMES, vars(), LOG_DIR / "options.json")

    # Create class dirs
    if OUTPUT_DIR_MODE == "dataset_dir":
        OUTPUT_DATASET_DIR.mkdir(exist_ok=True, parents=True)
        get_options_json(OPTION_NAMES, vars(), OUTPUT_DATASET_DIR / "options.json")
        for cl in bee.CLASSES:
            (OUTPUT_DATASET_DIR / cl).mkdir(exist_ok=True)

    # Find all trajectory dirs
    trajectory_dirs = [d for d in TRAJECTORIES_BASE_DIR.glob("*_trajectories*")]

    fragments_stats = []

    for trajectory_dir in trajectory_dirs:
        # Extract trajectory dir simple name
        try:
            scene_name = bee.dir_to_scene_name(trajectory_dir.name)
        except RuntimeError:
            print("### Skipping", trajectory_dir.name, ", Doesnt match file scene dir pattern!")
            continue

        scene_id = bee.scene_name_to_id(scene_name)
        scene_short_id = bee.scene_short_id_by_id(scene_id)

        # Find all files from directory
        trajectory_files = [file for file in trajectory_dir.iterdir() \
                     if (file.is_file() and file.name.endswith(".csv") and file.stat().st_size >= MIN_FILE_SIZE_BYTES)]

        print(f"\nSCENE {scene_id} containing {len(trajectory_files)} trajectories >= {MIN_FILE_SIZE_BYTES//1024}KB ####")

        # Iterate over files in directory
        for trajectory_filepath in trajectory_files:
            filestem = trajectory_filepath.name.replace(".csv", "")
            name_arr = filestem.split("_")
            instance_id = name_arr[0]
            cla = name_arr[1]
            clas = bee.parse_full_class_name(cla, "unknown")
            traj_event_count = int(name_arr[2][3:])
            traj_start_ts = int(name_arr[3][5:])

            print(f"├─ TRAJECTORY {scene_id}:{instance_id} ({clas}) ({traj_event_count}pts)")

            fragments = []
            
            with open(trajectory_filepath, 'r') as csv_file:
                reader = csv.reader(csv_file)
                # skip header
                csv_header = next(reader)

                EVENTS_CSV_P_COL = index_or_default(csv_header, "p", None)
                EVENTS_CSV_IS_CONFIDENT_COL = index_or_default(csv_header, "is_confident", None)
                EVENTS_CSV_FRAME_INDEX_COL = index_or_default(csv_header, "bb_frame_index", None)
                EVENTS_CSV_BB_CORNER_COL = index_or_default(csv_header, "bb_corner_index", None)
                
                start = 0
                next_fragment_start = start + T_BUCKET_LENGTH

                current_fragment = Fragment()
                current_fragment.start = 0
                current_fragment.index = 0

                for event_index, row in enumerate(reader):
                    # convert t back to original scale
                    t = float(row[EVENTS_CSV_T_COL]) / T_SCALE
                    x = int(row[EVENTS_CSV_X_COL])
                    y = int(row[EVENTS_CSV_Y_COL])

                    if t >= next_fragment_start:
                        # ts is in next fragment
                        # Save fragment
                        fragments.append(current_fragment)
                        # Start new fragment
                        current_fragment = Fragment()
                        current_fragment.start = t - (t % T_BUCKET_LENGTH)
                        current_fragment.index = int(t // T_BUCKET_LENGTH)
                        next_fragment_start = current_fragment.start + T_BUCKET_LENGTH
                        # print(f"Started new fragment {current_fragment.index} at {current_fragment.start/TIMESTEPS_PER_SECOND: >6.2f}; Next at {next_fragment_start/TIMESTEPS_PER_SECOND: >6.2f}")
                    # shift t so that the start of a fragment is at 0
                    reorigined_t = t - current_fragment.start
                    new_row = [x,y,reorigined_t, *(row[3:])]
                    # new_row[EVENTS_CSV_X_COL] = x
                    # new_row[EVENTS_CSV_Y_COL] = y
                    # new_row[EVENTS_CSV_T_COL] = reorigined_t
                    current_fragment.events.append( new_row )

                # -> If there are no events in the time frame of a fragment, the fragment wont be created. Is that a problem?

            # print("Fragmented trajectory into", len(fragments), "fragments")

            orig_fragment_count = len(fragments)
            upsampled_count = 0
            downsampled_count = 0
            upsampled_count = 0

            # Throw away fragments with too few events
            if MIN_EVENTS_COUNT is not None:
                fragments = [f for f in fragments if len(f.events) >= MIN_EVENTS_COUNT]

            for fragment in fragments:
                fragment.original_event_count = len(fragment.events)
                fragment.events_df = pd.DataFrame(data=fragment.events, columns=csv_header)
                # fragment.events = np.array(fragment.events)

                # t is currently unscaled. Scale it again
                fragment.start *= T_SCALE
                # fragment.events[:,2] *= T_SCALE
                fragment.events_df["t"] = fragment.events_df["t"].apply(lambda x: x*T_SCALE)

                if EVENTS_CSV_BB_CORNER_COL is not None:
                    bb_events_df = fragment.events_df.loc[fragment.events_df["bb_corner_index"].astype(int) >= 0]
                    fragment.events_df = fragment.events_df.loc[fragment.events_df["bb_corner_index"].astype(int) == -1]
                    fragment.original_event_count -= len(bb_events_df.index)
                
                # Add or remove points by sampling to get exactly EVENTS_PER_FRAGMENT points
                if EVENTS_PER_FRAGMENT != "all":
                    # Create open3d point cloud
                    fragment.events_df = fragment.events_df[["x","y","t"]]
                    xyz = fragment.events_df.to_numpy(dtype=np.float32)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz)
                    
                    # Noise reduction
                    if NOISE_REDUCTION_METHOD == "sor":
                        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=6.0)
                    elif NOISE_REDUCTION_METHOD == "radius":
                        pcd, ind = pcd.remove_radius_outlier(10, 16.0)
                    fragment.noise_reduced_event_count = len(pcd.points)

                    # remove points if number of points is > EVENTS_PER_TRAJECTORY
                    if len(pcd.points) > EVENTS_PER_FRAGMENT:
                        # Downsample
                        if DOWNSAMPLE_METHOD == "random":
                            # FIRST convert point cloud  back to pandas df
                            fragment.events_df = pd.DataFrame(data=np.array(pcd.points), columns=["x","y","t"])
                            # Use random sampling to choose subset of events
                            fragment.events_df = fragment.events_df.sample(n=EVENTS_PER_FRAGMENT, replace=False, random_state=0)
                        elif DOWNSAMPLE_METHOD == "farthest_point":
                            pcd_fps = pcd.farthest_point_down_sample(EVENTS_PER_FRAGMENT)
                            # BUG: farthest_point_down_sample sometimes returns less than N points!
                            if len(pcd_fps.points) == EVENTS_PER_FRAGMENT:
                                fragment.events_df = pd.DataFrame(data=np.array(pcd_fps.points), columns=["x","y","t"])
                            else:
                                # Fall back to random sampling
                                print("│ │ └─ WARNING: sampled_event_count != EVENTS_PER_FRAGMENT! Using random sampling!")
                                # FIRST convert point cloud  back to pandas df
                                fragment.events_df = pd.DataFrame(data=np.array(pcd.points), columns=["x","y","t"])
                                # Use random sampling to choose subset of events
                                fragment.events_df = fragment.events_df.sample(n=EVENTS_PER_FRAGMENT, replace=False, random_state=0)
                        else:
                            raise NotImplementedError("DOWNSAMPLE_METHOD " + str(DOWNSAMPLE_METHOD) + " not implemented!")
                        downsampled_count += 1

                    # add points if number of points is < EVENTS_PER_TRAJECTORY
                    elif len(pcd.points) < EVENTS_PER_FRAGMENT:
                        # Upsample
                        # FIRST Convert back to pandas df
                        fragment.events_df = pd.DataFrame(data=np.array(pcd.points), columns=["x","y","t"])
                        # frag_df = pd.DataFrame(data=fragment.events, columns=["x","y","t"])
                        # Use random sampling to duplicate as many existing events as are required to achive EVENTS_PER_TRAJECTORY. 
                        # Samples can be chosen multiple times
                        additional_df = fragment.events_df.sample(n=EVENTS_PER_FRAGMENT-len(fragment.events_df), replace=True, random_state=0)
                        # combine original events and created events
                        combined_df = pd.concat([fragment.events_df, additional_df])
                        fragment.events_df = combined_df
                        upsampled_count += 1

                fragment.sampled_event_count = len(fragment.events_df.index)

                # shuffle events or sort by time
                if SHUFFLE_T:
                    fragment.events_df = fragment.events_df.sample(frac=1).reset_index(drop=True)
                else:
                    fragment.events_df = fragment.events_df.sort_values("t")

                if NORMALIZE:
                    pc = fragment.events_df.to_numpy()
                    pc[:,:3] = pc_normalize(fragment.events_df.to_numpy()[:,:3])
                    fragment.events_df = pd.DataFrame(data=pc, columns=["x","y","t"])

                # Columns: 
                # scene, instance_id, fragment_id, class, traj_event_count, traj_start_ts, 
                # orig_event_count, nr_event_count, sampled_event_count
                fragments_stats.append( [scene_id, int(instance_id), fragment.index, clas, traj_event_count, traj_start_ts, \
                                         fragment.original_event_count, fragment.noise_reduced_event_count, fragment.sampled_event_count] )

                print(f"│ ├─ Fragment {fragment.index}: orig_evts={fragment.original_event_count} "\
                      + f"nr_evt_count={fragment.noise_reduced_event_count} sampled_evt_count={fragment.sampled_event_count}")
                
                if fragment.sampled_event_count != EVENTS_PER_FRAGMENT:
                    raise RuntimeError("original_event_count != EVENTS_PER_FRAGMENT")
                
            print(f"│ └─ Trajectory {instance_id}: orig_fragments={orig_fragment_count} remaining_fragments={len(fragments)} "\
                  + f"upsampled={upsampled_count} downsampeld={downsampled_count}")

            # for i, fragment in enumerate(fragments):
            #     print(f"- Fragment {fragment.index: >3}: evts:{len(fragment.events): >5}, orig_evts:{fragment.original_event_count: >5}, start:{fragment.start/T_SCALE/TIMESTEPS_PER_SECOND: >5.2f}s")

            # Save fragmented trajectories as CSVs
            if OUTPUT_DIR_MODE == "parent_dir":
                # save in parent dir of trajectory
                saved_count = 0
                output_base_dir = trajectory_dir / f"fragments_time_{T_BUCKET_LENGTH_MS}ms_{EVENTS_PER_FRAGMENT}pts{DATETIME_STR_PREFIX}"
                output_dir = output_base_dir / f"{filestem}"
                output_dir.mkdir(parents=True, exist_ok=True)
                for i, fragment in enumerate(fragments):
                    output_path = output_dir / f"frag_{fragment.index}.csv"
                    fragment.events_df.to_csv(output_path, index=False, header=True, decimal='.', sep=',', float_format='%.3f')
                # print(f"Saved {saved_count} trajectory files in {output_base_dir}")
            elif OUTPUT_DIR_MODE == "dataset_dir":
                # save all fragments in a common dir
                saved_count = 0
                output_dir = OUTPUT_DATASET_DIR / clas
                # save fragment files
                for i, fragment in enumerate(fragments):
                    # identify fragment with scene_name + instance_id + fragment_index
                    output_path = output_dir / f"{clas}_{scene_short_id}_{instance_id}_{fragment.index}.csv"
                    fragment.events_df.to_csv(output_path, index=False, header=True, decimal='.', sep=',', float_format='%.3f')
            else:
                raise RuntimeError("Invalid OUTPUT_DIR_MODE " + str(OUTPUT_DIR_MODE))

    if SAVE_STATISTICS:
        fragments_df = pd.DataFrame(fragments_stats, \
                columns=["scene", "instance_id", "fragment_id", "class", "traj_event_count", "traj_start_ts", \
                            "orig_event_count", "nr_event_count", "sampled_event_count"])
        fragments_df.to_csv(LOG_DIR/"fragments.csv", index=False, header=True, decimal='.', sep=',', float_format='%.3f')
        if OUTPUT_DIR_MODE == "dataset_dir":
            fragments_df.to_csv(LOG_DIR/"fragments.csv", index=False, header=True, decimal='.', sep=',', float_format='%.3f')

    print("Finished!")





    







                













