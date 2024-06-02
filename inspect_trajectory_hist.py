# Purpose: Visualize and export trajectory statistics.

import re
import csv
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_projected_heatmap(df, col1, col2, bins_x, bins_y):
    df_proj = df.loc[:,[col1, col2]]

    t_max = df_proj[col1].iloc[-1]
    # [[xmin, xmax], [ymin, ymax]]
    ty_hist_range = [ [0, int(t_max)], [0, bins_y] ]

    heatmap, xedges, yedges = np.histogram2d(df_proj[col1], df_proj[col2], bins=[bins_x, bins_y], density=False, range=ty_hist_range)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    heatmap = np.log2(heatmap+1)

    return df_proj, heatmap, extent


############################ MAIN ##################################
if __name__ == "__main__":
    np.set_printoptions(suppress=True,precision=3)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    PRINT_PROGRESS_EVERY_N_PERCENT = 1

    ADD_TIME_TO_FILENAME = True

    ############### PF #############
    WIDTH = 1280
    HEIGHT = 720
    FPS = 60

    # Format from DarkLabel software: (frame_index, classname, instance_id, is_difficult, x, y, w, h) 
    #                                      0            1          2            3         4  5  6  7

    # Format for events: (x, y, t (us or ms), p)
    EVENTS_CSV_X_COL = 0
    EVENTS_CSV_Y_COL = 1
    EVENTS_CSV_T_COL = 2
    EVENTS_CSV_P_COL = 3

    # Paths
    TRAJECTORIES_CSV_DIR = Path("output/extracted_trajectories")
    FIGURE_OUTPUT_DIR = Path("output/figures/projection_and_hist")

    EVENTS_CSV_HAS_HEADER = False
    LABELS_CSV_HAS_HEADER = False

    ##################################
    
    # Precision of the timestamp: for mikroseconds: 1000000, for milliseconds: 1000
    TIMESTEPS_PER_SECOND = 1_000_000
    # If timestamp in mikroseconds: -> mikroseconds per frame
    TIMESTEPS_PER_FRAME = (1 / FPS) * TIMESTEPS_PER_SECOND
    HALF_FRAME_TIME = TIMESTEPS_PER_FRAME // 2
    T_SCALE = 0.002 # 0.002 is good
    INV_T_SCALE = 1 / T_SCALE

    # Bsp: Biene hat 200 flügelschläge pro sek. Das sind 5ms pro flügelschlag.
    # Es sollen 10 Flügelschläge in ein Fenster passen. Also 5*10=50ms oder 50000us
    BUCKET_WIDTH_T = 1000 * 50

    # zb "1_l-l-l_trajectories_2024-05-29_15-27-12"
    dir_name_pattern = re.compile(r"^(.*)_trajectories")

    # Find existing figure dirs to skip them
    existing_figure_dir_names = [d.name for d in FIGURE_OUTPUT_DIR.glob("*_trajectories*")]

    # Find all files from directory; Skip existing
    trajectory_files = [f for f in TRAJECTORIES_CSV_DIR.glob("*_trajectories*/*.csv") if f.parent.name not in existing_figure_dir_names]

    # FOR TESTING!
    # trajectory_files = [TRAJECTORIES_CSV_DIR / "1_l-l-l_trajectories_2024-05-29_15-27-12/2.csv"]

    # Iterate over files in directory
    for trajectory_filepath in trajectory_files:
        trajectory_filestem = trajectory_filepath.name.replace(".csv", "")

        # Extract trajectory dir simple name
        matches = re.findall(dir_name_pattern, trajectory_filepath.parent.name)
        if len(matches) == 0:
            print("Skipping:", trajectory_filepath.parent.name, "/ trajectory:", trajectory_filestem, ". Doesnt match pattern!")
            continue
        trajectory_dir_name = matches[0]

        print("Processing:", trajectory_filepath.parent.name, "/ trajectory:", trajectory_filestem)

        df = pd.read_csv(trajectory_filepath, sep=',', header="infer")

        # timestamp in csv was multplied by 0.002 (or something else). Scale it back to real time (micros)
        t_col_real = df.loc[:,"t"] * INV_T_SCALE

        # t_max is scaled! t_max_real is in real time (e.g. micros)
        max_t_real = t_col_real.iloc[-1]
        max_t_str = f"{int((max_t_real / TIMESTEPS_PER_SECOND) // 60):0>2}m:{(max_t_real / TIMESTEPS_PER_SECOND % 60):0>2.2f}s."
        
        number_of_buckets = int(np.ceil(max_t_real/BUCKET_WIDTH_T))
        t_col_buckets = (t_col_real // BUCKET_WIDTH_T).astype('Int64')
        event_count_per_bucket = t_col_buckets.value_counts(sort=False).sort_index()
        # -> problem: if there are 0 events in a bucket, the bucket wont be included in the dataframe!
        event_count_per_bucket_no_gaps = pd.Series(data=[0]*number_of_buckets, index=list(range(number_of_buckets)))
        for x in event_count_per_bucket.items():
            event_count_per_bucket_no_gaps.at[x[0]] = x[1]
        event_count_per_bucket = event_count_per_bucket_no_gaps

        # print("max_t_real", max_t_real, max_t_str)
        # print("Die Bahn hat", number_of_buckets, "buckets")
        # print(t_col_buckets.head())
        # print(t_col_buckets.tail())
        # print(t_col_buckets.shape)
        # print(t_col_buckets.describe())
        # print(event_count_per_bucket.shape)
        # print(event_count_per_bucket.head())
        # print(event_count_per_bucket)

        # Project Trajectory to 2D plane: Only use (t,x) or (t,y)
        tx_df, tx_heatmap, tx_extent = get_projected_heatmap(df, "t", "x", 2000, WIDTH) # bins_x=number_of_buckets*10
        ty_df, ty_heatmap, ty_extent = get_projected_heatmap(df, "t", "y", 2000, HEIGHT)

        fig, axs = plt.subplots(3)
        fig.set_size_inches(20, 8)
        fig.suptitle(f't-x-projection, t-y-projection and event histogram of trajectory "{trajectory_dir_name}/{trajectory_filestem}" (length: {max_t_str})')
        axs[0].imshow(tx_heatmap.T, origin='lower', cmap="Greys", aspect="auto")
        axs[1].imshow(ty_heatmap.T, origin='lower', cmap="Greys", aspect="auto")
        axs[2].set_xlim(left=0, right=number_of_buckets)
        axs[2].bar(list(range(number_of_buckets)), event_count_per_bucket, color='navy', width=1.0, align="edge")

        output_dir = FIGURE_OUTPUT_DIR  / trajectory_filepath.parent.name
        output_dir.mkdir(exist_ok=True, parents=True)
        figure_filepath = output_dir / f"{trajectory_filestem}.png"
        plt.savefig(figure_filepath, bbox_inches='tight')
        plt.close()

