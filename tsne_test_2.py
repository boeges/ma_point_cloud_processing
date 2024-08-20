# use conda env "pix2pix"

from pathlib import Path
from datetime import datetime

import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Lasso
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

import bee_utils as bee



T_SCALE = 0.002
T_BUCKET_LENGTH = 1000 * 100 # 100ms
# Which samples of the dataset to show; "all", "train" or "test";
# Irrelevant for Autoencoder since its trained on all samples

ACTIVATIONS_DIR_PN = Path("../Pointnet_Pointnet2_pytorch/log/classification/")
ACTIVATIONS_DIR_FN = Path("../foldingnet2/snapshot/")

ACTIVATIONS_FILE = ACTIVATIONS_DIR_PN / "msg_cls5C_e40_bs8_pts4096_split7030_ds4/logs/activations_per_sample_2024-08-20_21-51.csv"
DATASET_DIR = Path("../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_4_w_depth")
DATASET_SPLIT_FILE = Path("../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_4/train_test_split_7030.txt")
# Contains exported 2d projections
FIGURES_DIR = Path("output/figures/projection_and_hist/tf100ms_tbr250_1")
# Labels mapping file will be exported to this dir
LABELS_OUTPUT_DIR = Path("output/instance_classes/tsne_inspector")
# Dir with bbox annotations; Existing files will be overwritten!
ANNOTATIONS_DIR = Path("output/video_annotations/3_classified")

SHOW_TEST_AS_UNLABELED = False
GROUP_TO_TRAJECORIES = True # True: Points are whole trajectories; false: Points are fragments


if __name__ == '__main__':
    # load activations file
    act_file_df = pd.read_csv(ACTIVATIONS_FILE, sep=",", header="infer")

    # Split df up in description df and activations df
    # description df:
    # Columns: sample_path, targex_index, target_name, orig_target_name, frag_id:tuple, scene_id, instance_id, fragment_index
    df = act_file_df[["sample_path", "target_name"]].copy()
    df["orig_target_name"] = df["target_name"].copy()
    # add "frag_id" column
    df.loc[:,"frag_id"] = df.loc[:,'sample_path'].apply(bee.frag_filename_to_id)
    # split up frag_id-tuple into 3 columns
    df[['scene_id', 'instance_id', "fragment_index"]] = pd.DataFrame(df['frag_id'].tolist(), index=df.index)
    # add "target_index" column
    classes_map = dict(zip(bee.CLASSES, range(len(bee.CLASSES))))
    df.loc[:,"target_index"] = df.loc[:,"target_name"].map(classes_map)

    if DATASET_SPLIT_FILE is not None:
        # find out split ("train"/"test") of each fragment
        # fid example: "hn-dra-1_16_6"
        train_fids, test_fids = bee.read_split_file(DATASET_SPLIT_FILE)
        df["split"] = df["frag_id"].apply(lambda frag_tuple: apply_split(frag_tuple, train_fids, test_fids))
    else:
        df["split"] = ""

    if SHOW_TEST_AS_UNLABELED:
        sel = df["split"]=="test"
        df.loc[sel,"orig_target_name"] = "insect"
        df.loc[sel,"target_name"] = "insect"
        df.loc[sel,"target_index"] = 1

    # activations / features
    # find range of activation columns: "act_0" - "act_N"
    max_act_col = [c for c in act_file_df.columns if str(c).startswith("act_")][-1]
    print("Using activation columns: act_0 -", max_act_col)
    # get all "act_X"-Columns
    act_df = act_file_df.loc[:,"act_0":max_act_col].copy()
    df = pd.concat([df, act_df], axis=1)

    # group fragments to trajecotries
    if GROUP_TO_TRAJECORIES:
        frags_df = df
        # Check if all intances have exactly one class.
        # If there are fragments of the same instance with different classes this will return false
        all_have_one_class = (df.groupby(["scene_id", "instance_id"]).nunique()["target_name"] == 1).all()
        if not all_have_one_class:
            print("ERROR: Cannot group by scene and instance_id; Some instances have more than one class!")
            exit()

        # Get the list of all activations columns
        act_cols = df.columns[df.columns.str.startswith("act_")]
        # from these columns take the first value in a group
        take_first_cols = ["frag_id", "sample_path","target_name","orig_target_name","target_index","split"]

        # Create an aggregation dictionary
        agg_dict = {}
        agg_dict.update( {col: "first" for col in take_first_cols} )
        agg_dict.update( {col: "mean" for col in act_cols} )

        # Group by column "A", aggregate the numeric columns by mean, and take the first value of "B"
        df = df.groupby(["scene_id","instance_id"]).agg(agg_dict).reset_index()

        # df = df.groupby(["scene_id","instance_id","target_name","orig_target_name","target_index","split"]).mean().drop(["fragment_index"], axis=1).reset_index()
        # df["frag_id"] = ""
        # df["sample_path"] = ""

        print(df.head())

    full_df = df

    # Find usd classes
    df_classes = full_df["orig_target_name"].unique()
    used_classes = [cl for cl in bee.CLASSES if cl in df_classes] # to keep order


    # if split != "all":
    #     df = df.drop(df[df["split"] != split].index)
    #     df.reset_index(drop=True, inplace=True)

    # Create plot
    subplot_kw = dict(xlim=(-10, 10), ylim=(-10, 10), autoscale_on=False)
    gridspec_kw = dict(height_ratios=[100,20,20])
    fig, (ax_tsne, ax_tx, ax_ty) = plt.subplots(3, 1, figsize=(6, 8), gridspec_kw=gridspec_kw)

    # array of tuple to 2d-array
    # fc = np.array([*(bee.get_rgba_of_class_index(descr_df["target_index"]).to_numpy())])

    point_size = TsneInspector.point_sizes[point_size_index]
    scatter = ax_tsne.scatter(x=get_df().loc[:,"tsne_x"], y=get_df().loc[:,"tsne_y"], \
                        s=point_size, picker=True, pickradius=10, linewidth=1.5) # c=df["target_index"], cmap="Set1", vmin=0, vmax=8, c=fc

    update_colors()
    
    ax_tsne.set_title("t-SNE plot of last inner layer activations")
    ax_tx.set_title("t-x-projection")
    ax_ty.set_title("t-y-projection")

    for ax in (ax_tsne, ax_tx, ax_ty):
        ax.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False) 

    # Events
    fig.canvas.mpl_connect('pick_event', on_pick)
    # not needed right now
    # fig.canvas.mpl_connect('motion_notify_event', on_move)


    # create legend
    legend_handles = []
    for i,cla in enumerate(bee.CLASSES):
        if cla not in used_classes:
            continue
        clr = bee.get_rgba_of_class_index(i, 0.66)
        h = mpatches.Patch(color=clr, label=cla)
        legend_handles.append(h)
    ax_tsne.legend(handles=legend_handles)

    fig.tight_layout()













