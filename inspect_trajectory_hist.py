# Purpose: Visualize and export trajectory statistics.

import re
import csv
import cv2
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bee_utils as bee

# width_x is the end of the last bucket on the x axis (usually t axis)
def get_projected_heatmap(df, col1, col2, bins_x, bins_y, width_x):
    df_proj = df.loc[:,[col1, col2]]

    # t_max = df_proj[col1].iloc[-1]
    # [[xmin, xmax], [ymin, ymax]]
    ty_hist_range = [ [0, width_x], [0, bins_y] ]

    heatmap, xedges, yedges = np.histogram2d(df_proj[col1], df_proj[col2], bins=[bins_x, bins_y], density=False, range=ty_hist_range)
    heatmap = heatmap.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # heatmap = np.log2(heatmap+1)
    return df_proj, heatmap, extent

def hist2d_to_image(arr):
    max_val_ln = np.log(arr.max()+1)
    # log1p is ln(x+1)
    arr = np.log1p(arr) / max_val_ln # -> [0, 1]
    # make 0=white (=255), 1=black (=0)
    arr = ((arr * -1) + 1.0) * 255.0
    return arr.astype(int)

def event_count_to_color(count, good_count):
    if count >= good_count*2:
        return "navy"
    elif count >= good_count:
        return "royalblue"
    elif count >= good_count//2:
        return "orange"
    return "tomato"

# Return bbox indices of bbox around all non zero values
def arg_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.nonzero(rows)[0][[0, -1]]
    xmin, xmax = np.nonzero(cols)[0][[0, -1]]
    return ymin, ymax+1, xmin, xmax+1

# Because cv2.imwrite doe not support umlauts (ä,ü,ö) 
def cv2_imwrite(path, image):
    _, im_buf_arr = cv2.imencode(".png", image)
    im_buf_arr.tofile(path)

# image as monochrome image with [0,255] (255=white)
def draw_t_ticks_into_image(image, pixel_distance):
    image = image.copy()
    # shape: (height,width)
    h = image.shape[0]
    w = image.shape[1]
    for x in range(0, w, pixel_distance):
        part_id = x // pixel_distance
        t_s = part_id * T_BUCKET_LENGTH / 1_000_000
        # Grid over full height
        image[20:h-20,x] = 191
        # Top and bottom ticks
        image[0:20,x-1:x+1] = 0
        image[h-20:h,x-1:x+1] = 0
        # Add part index and time to ticks
        text = f"p:{part_id:0>3} t:{t_s:0>5.2f}s"
        position = (x+5, 20)  # (x, y) coordinates
        cv2.putText(image, text, position, FONT, FONT_SCALE, FONT_COLOR, thickness=1)
    return image

def y_crop_to_used_area(image, padding=0):
    image_height = image.shape[0]
    ymin, ymax, xmin, xmax = arg_bbox(image)
    ymin = max(0, ymin-padding)
    ymax = min(image_height-1, ymax+padding)
    return image[ymin:ymax,:]

############################ MAIN ##################################
if __name__ == "__main__":
    np.set_printoptions(suppress=True,precision=3)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    PRINT_PROGRESS_EVERY_N_PERCENT = 1

    ADD_TIME_TO_FILENAME = True
    OVERWRITE_EXISTING = True

    ############### PF #############
    WIDTH = 1280
    HEIGHT = 720

    # Format for events: (x, y, t (us or ms), p)
    EVENTS_CSV_X_COL = 0
    EVENTS_CSV_Y_COL = 1
    EVENTS_CSV_T_COL = 2
    EVENTS_CSV_P_COL = 3

    # Paths
    TRAJECTORIES_CSV_DIR = Path("output/extracted_trajectories")
    FIGURE_OUTPUT_DIR = Path("output/figures/projection_and_hist") / DATETIME_STR

    ##################################
    
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
    # for the tx, ty projection: Sub-bins per bucket on the t axis
    BINS_PER_T_BUCKET = 250 
    # For saving matplotlib images
    SAVE_IMAGE_DPI = 300
    # Whether to crop images of the full flight trajectories to their used y areas
    Y_CROP_FULL_TRAJ_IMAGES = True
    # Draw ticks, vertical lines and text
    DRAW_T_TICKS = True
    # For text of ticks
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_COLOR = 127  # BGR color (here, green)


    # zb "1_l-l-l_trajectories_2024-05-29_15-27-12"
    dir_name_pattern = re.compile(r"^(.*)_trajectories")

    # Find existing figure dirs to skip them
    if OVERWRITE_EXISTING:
        existing_figure_dir_names = []
    else:
        existing_figure_dir_names = [d.name for d in FIGURE_OUTPUT_DIR.glob("*_trajectories*")]

    # Find all trajectory dirs; Skip existing
    trajectory_dirs = [d for d in TRAJECTORIES_CSV_DIR.glob("*_trajectories*") if d.name not in existing_figure_dir_names]

    # FOR TESTING!
    # trajectory_files = [TRAJECTORIES_CSV_DIR / "1_l-l-l_trajectories_2024-05-29_15-27-12/2.csv"]

    for trajectory_dir in trajectory_dirs:
        # Extract trajectory dir simple name
        matches = re.findall(dir_name_pattern, trajectory_dir.name)
        if len(matches) == 0:
            print("Skipping:", trajectory_dir.name, ", Doesnt match file name pattern!")
            continue
        trajectory_dir_name = matches[0]

        # Find all files from directory; Skip existing
        trajectory_files = [file for file in trajectory_dir.iterdir() if file.is_file()]

        print(f"\n#### Processing scene: \"{trajectory_dir.name}\" containing {len(trajectory_files)} trajectories ####")

        # Iterate over files in directory
        for trajectory_filepath in trajectory_files:
            filestem = trajectory_filepath.name.replace(".csv", "")
            name_arr = filestem.split("_")
            instance_id = name_arr[0]
            cla = name_arr[1]
            clas = bee.parse_full_class_name(cla, "insect")
            pts = int(name_arr[2][3:])
            start_ts = int(name_arr[3][5:])

            print(f"Processing: \"{trajectory_dir_name}/{instance_id}\" ({clas}) ({pts} points)")

            df = pd.read_csv(trajectory_filepath, sep=',', header="infer")

            # timestamp in csv was multplied by 0.002 (or something else). Scale it back to real time (micros)
            t_col_real = df.loc[:,"t"] * INV_T_SCALE

            # t_max is scaled! t_max_real is in real time (e.g. micros)
            max_t_real = t_col_real.iloc[-1]
            max_t_str = f"{int((max_t_real / TIMESTEPS_PER_SECOND) // 60):0>2}m:{(max_t_real / TIMESTEPS_PER_SECOND % 60):0>2.2f}s"
            
            number_of_buckets = int(np.ceil(max_t_real/T_BUCKET_LENGTH))
            bucket_index_per_event = (t_col_real // T_BUCKET_LENGTH).astype('Int64')
            event_count_per_bucket = bucket_index_per_event.value_counts(sort=False).sort_index()
            # -> problem: if there are 0 events in a bucket, the bucket wont be included in the dataframe!
            # create series of zeros with same length as event_count_per_bucket
            event_count_per_bucket_no_gaps = pd.Series(data=[0]*number_of_buckets, index=list(range(number_of_buckets)))
            for x in event_count_per_bucket.items():
                # x[0] is the bucket index, x[1] is the bucket event count
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

            # colors for bars in events histogram
            bar_colors = []
            for x in event_count_per_bucket:
                bar_colors.append(event_count_to_color(x, 4096))

            # Project Trajectory to 2D plane: Only use (t,x) or (t,y)
            tx_df, tx_heatmap, tx_extent = get_projected_heatmap(df, "t", "x", number_of_buckets*BINS_PER_T_BUCKET, WIDTH, number_of_buckets*T_BUCKET_LENGTH*T_SCALE)
            ty_df, ty_heatmap, ty_extent = get_projected_heatmap(df, "t", "y", number_of_buckets*BINS_PER_T_BUCKET, HEIGHT, number_of_buckets*T_BUCKET_LENGTH*T_SCALE)

            # log_tx_heatmap = np.log2(tx_heatmap+1)
            # log_ty_heatmap = np.log2(ty_heatmap+1)

            if Y_CROP_FULL_TRAJ_IMAGES:
                # Crop on y axis to used area
                tx_heatmap = y_crop_to_used_area(tx_heatmap, 50)
                ty_heatmap = y_crop_to_used_area(ty_heatmap, 50)

            tx_heatmap_image = hist2d_to_image(tx_heatmap)
            ty_heatmap_image = hist2d_to_image(ty_heatmap)

            if DRAW_T_TICKS:
                tx_heatmap_image = draw_t_ticks_into_image(tx_heatmap_image, BINS_PER_T_BUCKET)
                ty_heatmap_image = draw_t_ticks_into_image(ty_heatmap_image, BINS_PER_T_BUCKET)

            fig, axs = plt.subplots(3, gridspec_kw={'height_ratios': [1280,720,500]})
            fig.set_size_inches(20, 8)
            fig.suptitle(f't-x-, t-y-projection and event histogram of "{trajectory_dir_name}/{instance_id}" ({clas}) (length: {max_t_str}, {pts} points)')
            axs[0].imshow(tx_heatmap_image, origin='upper', cmap="gray", aspect="auto")
            axs[1].imshow(ty_heatmap_image, origin='upper', cmap="gray", aspect="auto")
            axs[2].set_xlim(left=0, right=number_of_buckets)
            axs[2].bar(list(range(number_of_buckets)), event_count_per_bucket, width=1.0, align="edge", color=bar_colors)

            # Save images of full trajectory
            tra_output_dir = FIGURE_OUTPUT_DIR  / trajectory_filepath.parent.name
            tra_output_dir.mkdir(exist_ok=True, parents=True)

            # Save detailled image
            figure_filepath = tra_output_dir / "detailled" / f"{filestem}_detailled.png"
            figure_filepath.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(figure_filepath, format="png", bbox_inches="tight", dpi=SAVE_IMAGE_DPI)
            plt.close()

            # Save tx-projection and ty-projection with OpenCV
            figure_filepath = tra_output_dir / "txproj" / f"{filestem}_txproj.png"
            figure_filepath.parent.mkdir(exist_ok=True, parents=True)
            cv2_imwrite(figure_filepath, tx_heatmap_image)

            figure_filepath = tra_output_dir / "typroj" / f"{filestem}_txproj.png"
            figure_filepath.parent.mkdir(exist_ok=True, parents=True)
            cv2_imwrite(figure_filepath, ty_heatmap_image)


            # Save images of full trajectory
            parts_output_dir_txproj = tra_output_dir / "parts" / "txproj"
            parts_output_dir_txproj.mkdir(exist_ok=True, parents=True)

            parts_output_dir_typroj = tra_output_dir / "parts" / "typroj"
            parts_output_dir_typroj.mkdir(exist_ok=True, parents=True)

            # Create images of parts of the trajectory
            for bucket_index in range(number_of_buckets-1):
                t_start = BINS_PER_T_BUCKET*bucket_index
                t_length = BINS_PER_T_BUCKET
                t_length_real = (t_length / BINS_PER_T_BUCKET) * T_BUCKET_LENGTH
                t_length_str = f"{(t_length_real / TIMESTEPS_PER_SECOND):0>2.2f}s"
                t_end = t_start+t_length
                event_count = event_count_per_bucket[bucket_index]

                if event_count == 0:
                    continue

                # Crop on t axis
                tx_heatmap_tcrop = tx_heatmap[:,t_start:t_end]
                ty_heatmap_tcrop = ty_heatmap[:,t_start:t_end]

                # Crop on y axis to used area
                ymin, ymax, xmin, xmax = arg_bbox(tx_heatmap_tcrop)
                tx_heatmap_tycrop = tx_heatmap_tcrop[ymin:ymax,:]

                # Crop on y axis to used area
                ymin, ymax, xmin, xmax = arg_bbox(ty_heatmap_tcrop)
                ty_heatmap_tycrop = ty_heatmap_tcrop[ymin:ymax,:]

                tx_heatmap_tycrop = hist2d_to_image(tx_heatmap_tycrop)
                ty_heatmap_tycrop = hist2d_to_image(ty_heatmap_tycrop)

                # Save tx-projection and ty-projection with OpenCV
                figure_filepath = parts_output_dir_txproj / f"{instance_id}_p{bucket_index}_txproj_{event_count}pts.png"
                cv2_imwrite(figure_filepath, tx_heatmap_tycrop)

                figure_filepath = parts_output_dir_typroj / f"{instance_id}_p{bucket_index}_typroj_{event_count}pts.png"
                cv2_imwrite(figure_filepath, ty_heatmap_tycrop)

        #         break
        #     break
        # break
