# Purpose: Fragment individual flight trajectories into smaller parts,
# so that they will fit into PointNet which requires 4096 points.
# Each part has the same time length. 
# Points will be reduced or created to achieve exactly 4096 points.

# TODO move points in (x,y) around origin?

import csv
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


class Fragment:
    def __init__(self):
        self.index = None
        self.start = None
        self.original_event_count = None
        # original events
        self.events = []
        # processed events as dataframe
        self.events_df = None

def indexOrDefault(l, value, default):
    try:
        return l.index(value)
    except ValueError:
        return default

############################ MAIN ##################################
if __name__ == "__main__":
    DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ADD_TIME_TO_FILENAME = True
    DATETIME_STR_PREFIX = ('_'+DATETIME_STR) if ADD_TIME_TO_FILENAME else ''

    PRINT_PROGRESS_EVERY_N_PERCENT = 1

    OVERWRITE_EXISTING = True

    ############### PF #############
    WIDTH = 1280
    HEIGHT = 720

    # Format for events: (x, y, t (us or ms), p)
    EVENTS_CSV_X_COL = 0
    EVENTS_CSV_Y_COL = 1
    EVENTS_CSV_T_COL = 2
    EVENTS_CSV_P_COL = None
    EVENTS_CSV_IS_CONFIDENT_COL = None
    EVENTS_CSV_FRAME_INDEX_COL = None
    EVENTS_CSV_BB_CORNER_COL = None

    # Paths
    # TRAJECTORIES_CSV_DIR = Path("output/extracted_trajectories") / "2_separated_2024-06-09_14-46-59" / "_with_bboxes"
    TRAJECTORIES_CSV_DIR = Path("output/extracted_trajectories") / "3_classified_pf1" / "_with_bboxes"


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
    # T_BUCKET_LENGTH = 1000 * 100
    T_BUCKET_LENGTH = 1000 * 4000 # 4s
    T_BUCKET_LENGTH_MS = int(T_BUCKET_LENGTH / TIMESTEPS_PER_SECOND * 1000)
    # This is our goal
    EVENTS_PER_FRAGMENT = 4096
    # Min number of events a fragment needs before adding or removing events
    MIN_EVENTS_COUNT = EVENTS_PER_FRAGMENT // 4

    # If the csv contains a EVENTS_CSV_BB_CORNER_COL, 
    # add bbox evemts to the resulting fragments or remove those events
    ADD_BB_EVENTS = False
    SAVE_PARTIAL_TRAJECTORIES = True
    MIN_FILE_SIZE_BYTES = 100 * 1024 # 100KB

    print(f"Using MAX_EVENTS_PER_TRAJECTORY={EVENTS_PER_FRAGMENT}, T_BUCKET_LENGTH_MS={T_BUCKET_LENGTH_MS}")

    scenes = [
        # "1_l-l-l",
        # "2_l-h-l",
        # "3_m-h-h",
        # "4_m-m-h",
        # "5_h-l-h",
        # "6_h-h-h_filtered",
        "hauptsächlichBienen1",
        # "hauptsächlichBienen1_trajectories_2024-06-03_19-41-13",
        # "libellen1_trajectories_2024-06-03_19-41-13",
        # "vieleSchmetterlinge2_trajectories_2024-06-03_19-41-13",
        # "wespen1_trajectories_2024-06-03_19-41-13",
        # "wespen2_trajectories_2024-06-03_19-41-13"
    ]

    for scene in scenes:
        csv_dir = TRAJECTORIES_CSV_DIR / f"{scene}_trajectories_bbox"
        output_base_dir = csv_dir / f"fragments_time_{T_BUCKET_LENGTH_MS}ms_{EVENTS_PER_FRAGMENT}pts{DATETIME_STR_PREFIX}"

        print(f"\nProcessing dir: {csv_dir}")
        
        # Find all csv files
        csv_paths = [csv_dir/file.name for file in csv_dir.iterdir() if (file.is_file() and file.name.endswith(".csv") and file.stat().st_size >= MIN_FILE_SIZE_BYTES)]
        print(f"Found {len(csv_paths)} files with >= {MIN_FILE_SIZE_BYTES//1024}KB")

        csv_header = None

        # TESTING: only use this file!
        # csv_paths = [csv_dir/"3_bee_pts38695_start7674867.csv"]
        
        for csv_path in csv_paths:
            print("Fragmenting trajectory file", csv_path, "...")
            
            fragments = []
            
            with open(csv_path, 'r') as csv_file:
                reader = csv.reader(csv_file)
                # skip header
                csv_header = next(reader)

                EVENTS_CSV_P_COL = indexOrDefault(csv_header, "p", None)
                EVENTS_CSV_IS_CONFIDENT_COL = indexOrDefault(csv_header, "is_confident", None)
                EVENTS_CSV_FRAME_INDEX_COL = indexOrDefault(csv_header, "bb_frame_index", None)
                EVENTS_CSV_BB_CORNER_COL = indexOrDefault(csv_header, "bb_corner_index", None)
                
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

            # Throw away fragments with too few events
            orig_fragment_count = len(fragments)
            fragments = [f for f in fragments if len(f.events) >= MIN_EVENTS_COUNT]
            # print(f"Filtered out {orig_fragment_count-len(fragments)} fragments with too few events")

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

                # remove points if number of points is > EVENTS_PER_TRAJECTORY
                if fragment.original_event_count > EVENTS_PER_FRAGMENT:
                    # frag_df = pd.DataFrame(data=fragment.events, columns=["x","y","t"])
                    # Use random sampling to choose subset of events
                    reduced_df = fragment.events_df.sample(n=EVENTS_PER_FRAGMENT, replace=False, random_state=0)
                    fragment.events_df = reduced_df

                # add points if number of points is < EVENTS_PER_TRAJECTORY
                elif fragment.original_event_count < EVENTS_PER_FRAGMENT:
                    # frag_df = pd.DataFrame(data=fragment.events, columns=["x","y","t"])
                    # Use random sampling to duplicate as many existing events as are required to achive EVENTS_PER_TRAJECTORY. 
                    # Samples can be chosen multiple times
                    additional_df = fragment.events_df.sample(n=EVENTS_PER_FRAGMENT-len(fragment.events_df), replace=True, random_state=0)
                    # combine original events and created events
                    combined_df = pd.concat([fragment.events_df, additional_df])
                    fragment.events_df = combined_df

                if EVENTS_CSV_BB_CORNER_COL is not None and ADD_BB_EVENTS:
                    # concat normal events and bbox events
                    fragment.events_df = pd.concat([fragment.events_df, bb_events_df])
                    # sort by t, then bb_corner_index
                    fragment.events_df = fragment.events_df.sort_values(["t", "bb_corner_index"])
                else:
                    # sort by t
                    fragment.events_df = fragment.events_df.sort_values("t")


            print(f"Fragmented into {orig_fragment_count: >2} fragments; {len(fragments): >2} have enough events")

            # for i, fragment in enumerate(fragments):
            #     print(f"- Fragment {fragment.index: >3}: evts:{len(fragment.events): >5}, orig_evts:{fragment.original_event_count: >5}, start:{fragment.start/T_SCALE/TIMESTEPS_PER_SECOND: >5.2f}s")

            # Save fragments
            # Create dir
            csv_stem = csv_path.name.replace(".csv", "")
            output_dir = output_base_dir / f"{csv_stem}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save fragmented trajectories as CSVs
            saved_count = 0
            for i, fragment in enumerate(fragments):
                output_path = output_dir / f"frag_{fragment.index}.csv"

                # with open(output_path, 'w', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerow(["x", "y", "t"])
                #     for event in fragment.events_df:
                #         # limit to n digits after decimal separator
                #         writer.writerow( event )
                #     saved_count += 1

                fragment.events_df.to_csv(output_path, index=False, header=True, decimal='.', sep=',', float_format='%.3f')

            print(f"Saved {saved_count} trajectory files in {output_base_dir}")

    print("Finished!")





    







                













