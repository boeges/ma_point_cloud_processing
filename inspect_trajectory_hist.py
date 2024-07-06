# Purpose: Visualize and export trajectory statistics.

import re
import csv
import cv2
import json
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
def draw_t_ticks_into_image(image, pixel_distance, predicitons=None, classes=None, scene_short_id=None, instance_id=None):
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
        cv2.putText(image, text, position, FONT_NAME, FONT_SCALE, FONT_COLOR, thickness=1)
        # write predicted class per fragment and confidence
        if predicitons != None:
            frag_key = (scene_short_id, int(instance_id), int(part_id))
            pred = predicitons.get(frag_key, None)
            if pred != None:
                target_name = pred["target_name"]
                target_conf = float(pred[target_name])
                pred_idx = int(pred["pred_choice"])
                pred_name = classes[pred_idx]
                pred_conf = float(pred[pred_name])
                font_color = FONT_COLOR if target_name==pred_name else 255
                cv2.putText(image, f"tgt:{target_name[:3].upper()} {target_conf: .2f}", (x+5, 40), FONT_NAME, FONT_SCALE, font_color, thickness=1)
                cv2.putText(image, f"prd:{pred_name[:3].upper()} {pred_conf: .2f}", (x+5, 60), FONT_NAME, FONT_SCALE, font_color, thickness=1)

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

    WIDTH = 1280
    HEIGHT = 720

    # Format for events: (x, y, t (us or ms), p)
    EVENTS_CSV_X_COL = 0
    EVENTS_CSV_Y_COL = 1
    EVENTS_CSV_T_COL = 2
    EVENTS_CSV_P_COL = 3
    
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
    T_BUCKET_LENGTH = 1000 * 100 # 100ms
    # T_BUCKET_LENGTH = 1000 * 4000 # 4s
    T_BUCKET_LENGTH_MS = int(T_BUCKET_LENGTH / 1000)
    # for the tx, ty projection: Sub-bins per bucket on the t axis
    # short: tbr: t bucket resolutions
    # BINS_PER_T_BUCKET = 250
    BINS_PER_T_BUCKET = int(T_BUCKET_LENGTH_MS * 2.5) # 250 for 100ms, 10000 for 4000ms
    
    SAVE_STATISTICS = False
    SAVE_IMAGES = True
    # BINS_PER_T_BUCKET = T_BUCKET_LENGTH_MS # a bucket for every ms
    # For saving matplotlib images
    SAVE_IMAGE_DPI = 300
    # Save images of each individual bucket (WARNING: Creates many images)
    SAVE_BUCKET_IMAGES = False
    # Whether to crop images of the full flight trajectories to their used y areas
    Y_CROP_FULL_TRAJ_IMAGES = False
    # Draw ticks, vertical lines and text
    DRAW_T_TICKS = True
    DRAW_PREDICTIONS = True
    DRAW_PREDICTIONS_STR = "_pred" if DRAW_PREDICTIONS else ""
    # For text of ticks
    FONT_NAME = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_COLOR = 127  # BGR color (here, green)


    # Paths
    TRAJECTORIES_BASE_DIR = Path("output/extracted_trajectories/3_classified")
    PREDICTION_FILE = Path("../Pointnet_Pointnet2_pytorch/log/classification/2024-07-03_23-11/logs/pred_per_class_2024-07-05_12-39.csv")
    # tf: timeframe
    FIGURE_OUTPUT_DIR = Path("output/figures/projection_and_hist") / f"tf{T_BUCKET_LENGTH_MS}ms_tbr{BINS_PER_T_BUCKET}{DRAW_PREDICTIONS_STR}_{DATETIME_STR}"
    STATS_OUTPUT_DIR = Path("output/statistics/hist/") / f"tf{T_BUCKET_LENGTH_MS}ms_{DATETIME_STR}"

    # stats = {
    # <scene_name>: {
    #   fields, 
    #   trajectories: {
    #       <traj_id>: {
    #           fields, 
    #           fragmentations: {
    #               <fragmentation>: {
    #                   fields, 
    #                   fragments: {
    #                       <frag_id>: {
    #                           fields
    #                       }
    #                   }
    #               }
    #           }
    #       }
    #   }
    # }}
    stats = {}
    fragments_stats = []

    # Read predicitons csv
    # key: (scene_id, instance_id, fragment_index)
    # data: pred_choice, pred value per class...
    # example: ('h3', 20, 4) {'pred_choice': '2', 'bee': '-3.4066', 'butterfly': '-1.6550', ...}
    predicitons = {}
    pred_classes = []
    if DRAW_PREDICTIONS:
        with open(PREDICTION_FILE, 'r') as input_labels_file:
            # column: sample_path, target_name, target_id, pred_choice, bee,butterfly,dragonfly,wasp,insect
            reader = csv.DictReader(input_labels_file)
            header = next(reader)
            hk = list(header.keys())
            first_class_index = hk.index("bee")
            pred_classes = hk[first_class_index:]

            for row in reader:
                # make key
                # example: "dragonfly\dragonfly_h3_6_5.csv".
                # get "h3_6_5" and split to "h3","6","5".
                # (scene_id, instance_id, fragment_index).
                # convert "6" and "5" to int.
                frag_id = row["sample_path"].replace(".csv","").split("_")[-3:]
                frag_id[1] = int(frag_id[1])
                frag_id[2] = int(frag_id[2])
                frag_id = tuple(frag_id)
                # add pred_choice and every class prediction to the dict
                pred = {}
                for k in ["target_name", "target_id", "pred_choice"] + pred_classes:
                    pred[k] = row[k]
                predicitons[frag_id] = pred

    # zb "1_l-l-l_trajectories_2024-05-29_15-27-12"
    dir_name_pattern = re.compile(r"^(.*)_trajectories.*")

    # Find existing figure dirs to skip them
    if OVERWRITE_EXISTING:
        existing_figure_dir_names = []
    else:
        existing_figure_dir_names = [d.name for d in FIGURE_OUTPUT_DIR.glob("*_trajectories*")]

    # Find all trajectory dirs; Skip existing
    trajectory_dirs = [d for d in TRAJECTORIES_BASE_DIR.glob("*_trajectories*") if d.name not in existing_figure_dir_names]

    # FOR TESTING!
    # trajectory_dirs = [TRAJECTORIES_CSV_DIR / "6_h-h-h_filtered_trajectories"]

    for trajectory_dir in trajectory_dirs:
        # Extract trajectory dir simple name
        try:
            scene_name = bee.dir_to_scene_name(trajectory_dir.name)
            scene_id = bee.scene_name_to_id(scene_name)
            scene_short_id = bee.scene_id_to_short_id(scene_id)
        except RuntimeError:
            print("Skipping:", trajectory_dir.name, ", Doesnt match file scene dir pattern!")
            continue

        stats.setdefault(scene_name, {"event_count": None, "length_s": None, "trajectories": {}})

        # Find all files from directory; Skip existing
        trajectory_files = [file for file in trajectory_dir.iterdir() if file.is_file()]

        print(f"\n#### Processing scene: \"{trajectory_dir.name}\" ({scene_id}) containing {len(trajectory_files)} trajectories ####")

        # Iterate over files in directory
        for trajectory_filepath in trajectory_files:
            filestem = trajectory_filepath.name.replace(".csv", "")
            name_arr = filestem.split("_")
            instance_id = name_arr[0]
            cla = name_arr[1]
            clas = bee.parse_full_class_name(cla, "insect")
            pts = int(name_arr[2][3:])
            start_ts = int(name_arr[3][5:])
            fragmentation_key = f"tf{T_BUCKET_LENGTH_MS}ms"

            traj_stats = stats[scene_name]["trajectories"].setdefault(int(instance_id), \
                    {"event_count": None, "length_s": None, "fragmentations": {}})
            fragmentation_stats = traj_stats["fragmentations"].setdefault(fragmentation_key, \
                    {"time_frame_ms": T_BUCKET_LENGTH_MS, "fragments": {}})

            print(f"Processing: \"{scene_name}/{instance_id}\" ({clas}) ({pts} points)")

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

            if SAVE_STATISTICS:
                STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                traj_stats["event_count"] = len(df)
                traj_stats["length_s"] = max_t_real / TIMESTEPS_PER_SECOND
                traj_stats["class"] = clas

                # Iterate fragments
                for bucket_index in range(number_of_buckets):
                    t_start = BINS_PER_T_BUCKET*bucket_index
                    t_length = BINS_PER_T_BUCKET
                    t_length_real = (t_length / BINS_PER_T_BUCKET) * T_BUCKET_LENGTH
                    t_end = t_start+t_length
                    event_count = event_count_per_bucket[bucket_index]

                    fragment_stats = fragmentation_stats["fragments"].setdefault(bucket_index, {})
                    # fragment_stats["t_start_s"] = t_start / TIMESTEPS_PER_SECOND
                    # fragment_stats["t_length_s"] = t_length_real
                    # fragment_stats["t_end_s"] = t_end / TIMESTEPS_PER_SECOND
                    fragment_stats["event_count"] = int(event_count)

                    # Columns: scene, instance_id, fragment_id, class, traj_evnt_count, traj_len_s, frag_evnt_count, frag_len_s
                    fragments_stats.append( [scene_name, int(instance_id), bucket_index, clas, len(df), traj_stats["length_s"], int(event_count), t_length_real/TIMESTEPS_PER_SECOND] )

            if SAVE_IMAGES:
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
                    tx_heatmap_ycrop = y_crop_to_used_area(tx_heatmap, 50)
                    ty_heatmap_ycrop = y_crop_to_used_area(ty_heatmap, 50)
                else:
                    tx_heatmap_ycrop = tx_heatmap
                    ty_heatmap_ycrop = ty_heatmap

                tx_heatmap_image = hist2d_to_image(tx_heatmap_ycrop)
                ty_heatmap_image = hist2d_to_image(ty_heatmap_ycrop)

                if DRAW_T_TICKS:
                    tx_heatmap_image = draw_t_ticks_into_image(tx_heatmap_image, BINS_PER_T_BUCKET, predicitons, pred_classes, scene_short_id, instance_id)
                    ty_heatmap_image = draw_t_ticks_into_image(ty_heatmap_image, BINS_PER_T_BUCKET, predicitons, pred_classes, scene_short_id, instance_id)

                fig, axs = plt.subplots(3, gridspec_kw={'height_ratios': [1280,720,500]})
                fig.set_size_inches(20, 8)
                fig.suptitle(f't-x-, t-y-projection and event histogram of "{scene_name}/{instance_id}" ({clas}) (length: {max_t_str}, {pts} points)')
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

                figure_filepath = tra_output_dir / "typroj" / f"{filestem}_typroj.png"
                figure_filepath.parent.mkdir(exist_ok=True, parents=True)
                cv2_imwrite(figure_filepath, ty_heatmap_image)

                # Save images of each individual bucket (WRNING: Creates many images)
                if SAVE_BUCKET_IMAGES:
                    parts_output_dir_txproj = tra_output_dir / "parts" / "txproj"
                    parts_output_dir_txproj.mkdir(exist_ok=True, parents=True)

                    parts_output_dir_typroj = tra_output_dir / "parts" / "typroj"
                    parts_output_dir_typroj.mkdir(exist_ok=True, parents=True)

                    # Create images of parts of the trajectory
                    for bucket_index in range(number_of_buckets):
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

    if SAVE_STATISTICS:
        with open(STATS_OUTPUT_DIR / "all.json", "w") as outfile:
            json.dump(stats, outfile)

        fragments_df = pd.DataFrame(fragments_stats, \
                columns=["scene", "instance_id", "fragment_id", "class", "traj_evnt_count", "traj_len_s", "frag_evnt_count", "frag_len_s"])
        fragments_df.to_csv(STATS_OUTPUT_DIR / "all_fragments.csv", index=False, header=True, decimal='.', sep=',', float_format='%.3f')
        print(fragments_df)
