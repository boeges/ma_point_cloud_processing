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

# Because cv2.imwrite doe not support umlauts (ä,ü,ö) 
def cv2_imwrite(path, image):
    _, im_buf_arr = cv2.imencode(".png", image)
    im_buf_arr.tofile(path)

# image as monochrome image with [0,255] (255=white)
def draw_t_ticks_into_image(image, pixel_distance, predicitons=None, classes=None, scene_id=None, instance_id=None):
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
            frag_key = (scene_id, int(instance_id), int(part_id))
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

# Return bbox indices of bbox around all non zero values
def arg_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.nonzero(rows)[0][[0, -1]]
    xmin, xmax = np.nonzero(cols)[0][[0, -1]]
    return ymin, ymax+1, xmin, xmax+1

def y_crop_to_used_area(image, padding=0):
    image_height = image.shape[0]
    ymin, ymax, xmin, xmax = arg_bbox(image)
    ymin = max(0, ymin-padding)
    ymax = min(image_height-1, ymax+padding)
    return image[ymin:ymax,:]


def calc_std_per_frag(df:pd.DataFrame, agg_type="min") -> pd.Series:
    """
    Calculate std of parts of a fragment, then take avg per fragment.

    Args:
        df (pd.DataFrame): df
        agg_type (str, optional): min or mean. Defaults to "min".

    Returns:
        pd.Series: Series with Std per Fragment
    """

    agg = {
        "x":"std",
        "y":"std",
    }

    df1 = df.groupby([df.fragment_index, df.stat_bucket]).agg(agg)

    # df1["std"] = df1[["x","y"]].sum(axis=1, min_count=1)
    df1["std"] = df1[["x","y"]].mean(axis=1)
    df1["std"].fillna(0.0, inplace=True)
    # df1["std"] = df1[["x","y"]].max(axis=1)

    agg1 = {
        "std":agg_type,
    }

    df1 = df1.groupby(level=0).agg(agg1)
    # df1.columns = df1.columns.get_level_values(1)
    # df1.rename(columns={"mean":"std_mean","min":"std_min","sum":"event_count"}, inplace=True)
    # display("mean STD per fragment over its subfragments", df1)

    return df1["std"]




############################ MAIN ##################################
if __name__ == "__main__":
    np.set_printoptions(suppress=True,precision=3)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    ADD_TIME_TO_FILENAME = True
    DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DATETIME_STR_PREFIX = ('_'+DATETIME_STR) if ADD_TIME_TO_FILENAME else ''

    PRINT_PROGRESS_EVERY_N_PERCENT = 1

    OVERWRITE_EXISTING = False

    WIDTH = 1280
    HEIGHT = 720

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
    STAT_BINS_PER_T_BUCKET = 20
    STAT_BIN_WIDTH = T_BUCKET_LENGTH / STAT_BINS_PER_T_BUCKET # in us

    
    # Save trajectory statistics
    SAVE_STATISTICS = True
    # Save images of complete paths
    SAVE_IMAGES = False
    # Save images of each individual bucket (WARNING: Creates many images)
    SAVE_BUCKET_IMAGES = False
    # BINS_PER_T_BUCKET = T_BUCKET_LENGTH_MS # a bucket for every ms
    # For saving matplotlib images
    SAVE_IMAGE_DPI = 300
    # Whether to crop images of the full flight trajectories to their used y areas
    Y_CROP_FULL_TRAJ_IMAGES = True
    # Draw ticks, vertical lines and text
    DRAW_T_TICKS = True
    DRAW_PREDICTIONS = False
    DRAW_PREDICTIONS_STR = "_pred" if DRAW_PREDICTIONS else ""
    # For text of ticks
    FONT_NAME = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_COLOR = 127  # BGR color (here, green)

    STD_AGG_TYPE = "min" # min or mean

    # Paths
    TRAJECTORIES_BASE_DIR = Path("output/extracted_trajectories/3_classified")
    PREDICTION_FILE = Path("../Pointnet_Pointnet2_pytorch/log/classification/msg_cls4A_e40_bs16_split7030/logs/pred_per_sample_2024-08-07_23-49.csv")
    # tf: timeframe
    FIGURE_OUTPUT_DIR = Path("output/figures/projection_and_hist") / f"tf{T_BUCKET_LENGTH_MS}ms_tbr{BINS_PER_T_BUCKET}{DRAW_PREDICTIONS_STR}{DATETIME_STR_PREFIX}"
    STATS_OUTPUT_DIR = Path("output/statistics/hist/") / f"tf{T_BUCKET_LENGTH_MS}ms_{DATETIME_STR}"

    # only include these scenes
    SCENE_STARTS_WITH = [
        # "hn-was",
        # "mb-bum"
    ]

    # scene name starts with
    EXCLUDE_SCENES_FROM_STATS = [
        "hn-depth",
    ]

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
                frag_id = bee.frag_filename_to_id(row["sample_path"])
                # add pred_choice and every class prediction to the dict
                pred = {}
                for k in ["target_name", "target_id", "pred_choice"] + pred_classes:
                    pred[k] = row[k]
                predicitons[frag_id] = pred

    # zb "1_l-l-l_trajectories_2024-05-29_15-27-12"
    # dir_name_pattern = re.compile(r"^(.*)")

    # Find existing figure dirs to skip them
    if OVERWRITE_EXISTING:
        existing_figure_dir_names = []
    else:
        existing_figure_dir_names = [d.name for d in FIGURE_OUTPUT_DIR.glob("*")]

    # Find all trajectory dirs; Skip existing
    trajectory_dirs = [d for d in TRAJECTORIES_BASE_DIR.glob("*") if d.name not in existing_figure_dir_names]

    # Skip scenes that do not start with a scene name in SCENE_STARTS_WITH
    for trajectory_dir in trajectory_dirs:
        if len(SCENE_STARTS_WITH) > 0:
            found_scene = False
            for sn in SCENE_STARTS_WITH:
                if trajectory_dir.name.startswith(sn):
                    found_scene = True
                    break
            if not found_scene:
                print("Skipping:", trajectory_dir.name, ". Scene name not in SCENE_STARTS_WITH!")
                continue

        exclude_scene_from_stats = False
        if len(EXCLUDE_SCENES_FROM_STATS) > 0:
            for sn in EXCLUDE_SCENES_FROM_STATS:
                if trajectory_dir.name.startswith(sn):
                    exclude_scene_from_stats = True
                    break
            if exclude_scene_from_stats:
                print("Excluding", trajectory_dir.name, "from stats!")

        # Extract trajectory dir simple name
        try:
            scene_id = bee.dir_to_scene_name(trajectory_dir.name)
            scene_id = bee.scene_name_to_id(scene_id)
        except RuntimeError as e:
            print("Skipping:", trajectory_dir.name, e)
            continue

        stats.setdefault(scene_id, {"event_count": None, "length_s": None, "trajectories": {}})

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

            traj_stats = stats[scene_id]["trajectories"].setdefault(int(instance_id), \
                    {"event_count": None, "length_s": None, "fragmentations": {}})
            fragmentation_stats = traj_stats["fragmentations"].setdefault(fragmentation_key, \
                    {"time_frame_ms": T_BUCKET_LENGTH_MS, "fragments": {}})

            print(f"Processing: \"{scene_id}/{instance_id}\" ({clas}) ({pts} points)")

            # Read all trajectory events
            df = pd.read_csv(trajectory_filepath, sep=',', header="infer")

            # timestamp in csv was multplied by 0.002 (or something else). Scale it back to real time (micros)
            df["t_real"] = df.loc[:,"t"] * INV_T_SCALE
            df["stat_bucket"]  = df.t_real.floordiv(STAT_BIN_WIDTH).astype('Int64')

            # t_max is scaled! t_max_real is in real time (e.g. micros)
            max_t_real = df["t_real"].iloc[-1]
            max_t_str = f"{int((max_t_real / TIMESTEPS_PER_SECOND) // 60):0>2}m:{(max_t_real / TIMESTEPS_PER_SECOND % 60):0>2.2f}s"
            
            df["fragment_index"] = (df["t_real"] // T_BUCKET_LENGTH).astype('Int64')
            number_of_fragments = int(np.ceil(max_t_real/T_BUCKET_LENGTH))

            ev_count_per_fragment = df["fragment_index"].value_counts(sort=False).sort_index()
            ev_count_per_fragment = ev_count_per_fragment.reindex(list(range(0, number_of_fragments)), fill_value=0)
            # -> problem: if there are 0 events in a bucket, the bucket wont be included in the dataframe!
            # create series of zeros with same length as event_count_per_bucket
            # ev_count_per_fragment_no_gaps = pd.Series(data=[0]*number_of_fragments, index=list(range(number_of_fragments)))
            # for x in ev_count_per_fragment.items():
            #     # x[0] is the bucket index, x[1] is the bucket event count
            #     ev_count_per_fragment_no_gaps.at[x[0]] = x[1]
            # ev_count_per_fragment = ev_count_per_fragment_no_gaps
            
            # print("max_t_real", max_t_real, max_t_str)
            # print("Die Bahn hat", number_of_buckets, "buckets")
            # print(t_col_buckets.head())
            # print(t_col_buckets.tail())
            # print(t_col_buckets.shape)
            # print(t_col_buckets.describe())
            # print(event_count_per_bucket.shape)
            # print(event_count_per_bucket.head())
            # print(event_count_per_bucket)

            # Calc stats (Std) per fragment
            std_per_fragment = calc_std_per_frag(df, agg_type=STD_AGG_TYPE)
            std_per_fragment = std_per_fragment.reindex(list(range(0, number_of_fragments)))
            

            if SAVE_STATISTICS and not exclude_scene_from_stats:
                STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                traj_stats["event_count"] = len(df)
                traj_stats["length_s"] = max_t_real / TIMESTEPS_PER_SECOND
                traj_stats["class"] = clas
                traj_stats["overall_mean_std"] = std_per_fragment.mean()
                traj_stats["overall_min_std"] = std_per_fragment.min()
                traj_stats["overall_max_std"] = std_per_fragment.max()

                # Iterate fragments
                for frag_index in range(number_of_fragments):
                    t_start = BINS_PER_T_BUCKET*frag_index
                    t_length = BINS_PER_T_BUCKET
                    t_length_real = T_BUCKET_LENGTH
                    frag_len_s = t_length_real/TIMESTEPS_PER_SECOND
                    t_end = t_start+t_length # ???
                    event_count = ev_count_per_fragment[frag_index]
                    traj_ev_cnt = len(df.index)
                    std = std_per_fragment[frag_index]

                    fragment_stats = fragmentation_stats["fragments"].setdefault(frag_index, {})
                    # fragment_stats["t_start_s"] = t_start / TIMESTEPS_PER_SECOND
                    # fragment_stats["t_length_s"] = t_length_real
                    # fragment_stats["t_end_s"] = t_end / TIMESTEPS_PER_SECOND
                    fragment_stats["event_count"] = int(event_count)

                    # Columns: scene, instance_id, fragment_id, class, traj_evnt_count, traj_len_s, frag_evnt_count, frag_len_s
                    fragments_stats.append( [scene_id, int(instance_id), frag_index, clas, traj_ev_cnt, traj_stats["length_s"], \
                                                int(event_count), frag_len_s, std] )

            if SAVE_IMAGES:
                # colors for bars in events histogram
                bar_colors = []
                for x in ev_count_per_fragment:
                    bar_colors.append(event_count_to_color(x, 4096))

                # Project Trajectory to 2D plane: Only use (t,x) or (t,y)
                tx_df, tx_heatmap, tx_extent = get_projected_heatmap(df, "t", "x", number_of_fragments*BINS_PER_T_BUCKET, WIDTH, number_of_fragments*T_BUCKET_LENGTH*T_SCALE)
                ty_df, ty_heatmap, ty_extent = get_projected_heatmap(df, "t", "y", number_of_fragments*BINS_PER_T_BUCKET, HEIGHT, number_of_fragments*T_BUCKET_LENGTH*T_SCALE)

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
                    tx_heatmap_image = draw_t_ticks_into_image(tx_heatmap_image, BINS_PER_T_BUCKET, predicitons, pred_classes, scene_id, instance_id)
                    ty_heatmap_image = draw_t_ticks_into_image(ty_heatmap_image, BINS_PER_T_BUCKET, predicitons, pred_classes, scene_id, instance_id)

                fig, axs = plt.subplots(4, gridspec_kw={'height_ratios': [1280,720,300,300]})
                fig.set_size_inches(20, 8)
                fig.suptitle(f't-x-, t-y-projection and event histogram of "{scene_id}/{instance_id}" ({clas}) (length: {max_t_str}, {pts} points)',\
                             size=14, y=0.92)

                axs[0].imshow(tx_heatmap_image, origin='upper', cmap="gray", aspect="auto")
                axs[1].imshow(ty_heatmap_image, origin='upper', cmap="gray", aspect="auto")

                # Show all x ticks if less than 40 fragments; Else its too much text
                xtick_label = ev_count_per_fragment.index if number_of_fragments <= 40 else None

                axs[2].set_xlim(left=0, right=number_of_fragments)
                axs[2].bar(ev_count_per_fragment.index, ev_count_per_fragment, width=0.98, align="edge", \
                           color=bar_colors, tick_label=xtick_label)
                axs[2].grid(axis="y")

                axs[3].set_xlim(left=0, right=number_of_fragments)
                axs[3].bar(x=std_per_fragment.index, height=std_per_fragment, width=0.98, align="edge", \
                           color="#2299cc", tick_label=xtick_label)
                axs[3].grid(axis="y")


                # Save images of full trajectory
                tra_output_dir = FIGURE_OUTPUT_DIR / trajectory_filepath.parent.name
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
                    for frag_index in range(number_of_fragments):
                        t_start = BINS_PER_T_BUCKET*frag_index
                        t_length = BINS_PER_T_BUCKET
                        t_length_real = (t_length / BINS_PER_T_BUCKET) * T_BUCKET_LENGTH
                        t_length_str = f"{(t_length_real / TIMESTEPS_PER_SECOND):0>2.2f}s"
                        t_end = t_start+t_length
                        event_count = ev_count_per_fragment[frag_index]

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
                        figure_filepath = parts_output_dir_txproj / f"{instance_id}_p{frag_index}_txproj_{event_count}pts.png"
                        cv2_imwrite(figure_filepath, tx_heatmap_tycrop)

                        figure_filepath = parts_output_dir_typroj / f"{instance_id}_p{frag_index}_typroj_{event_count}pts.png"
                        cv2_imwrite(figure_filepath, ty_heatmap_tycrop)

        #         break
        #     break
        # break

    if SAVE_STATISTICS:
        STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATS_OUTPUT_DIR / "all.json", "w") as outfile:
            json.dump(stats, outfile)

        fragments_df = pd.DataFrame(fragments_stats, \
                columns=["scene", "instance_id", "fragment_id", "class", "traj_evnt_count", "traj_len_s", "frag_evnt_count", "frag_len_s", "frag_std"])
        fragments_df.to_csv(STATS_OUTPUT_DIR / "all_fragments.csv", index=False, header=True, decimal='.', sep=',', float_format='%.3f')
        print(fragments_df)
