
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

# ACTIVATIONS_FILE = Path("../Pointnet_Pointnet2_pytorch/log/classification/2024-07-03_23-11/logs/activations_per_class_2024-07-23_12-30_with_bum.csv")
# ACTIVATIONS_FILE = Path("../Pointnet_Pointnet2_pytorch/log/classification/2024-07-25_22-10/logs/activations_per_class_2024-07-25_22-58.csv")
# ACTIVATIONS_FILE = Path("../Pointnet_Pointnet2_pytorch/log/classification/msg_cls6B_e40_bs8_split7030/logs/activations_per_sample_2024-08-03_22-36.csv")
# ACTIVATIONS_FILE = Path("../Pointnet_Pointnet2_pytorch/log/classification/msg_cls6B_e40_bs8_split2080/logs/activations_per_sample_2024-08-03_22-37.csv")
# ACTIVATIONS_FILE = Path("../Pointnet_Pointnet2_pytorch/log/classification/msg_cls4A_e40_bs8_split7030_3/logs/activations_per_sample_2024-08-09_09-27.csv")
# ACTIVATIONS_FILE = Path("../Pointnet_Pointnet2_pytorch/log/classification/msg_cls4A_e40_bs8_split20shot_3/logs/activations_per_sample_2024-08-09_11-36.csv")
# ACTIVATIONS_FILE = Path("../Pointnet_Pointnet2_pytorch/log/classification/msg_cls4A_e40_bs8_split40shot_3/logs/activations_per_sample_2024-08-09_19-50.csv")
ACTIVATIONS_FILE = Path("../Pointnet_Pointnet2_pytorch/log/classification/msg_cls4A_e40_bs8_pts4096_split40shot_4/logs/activations_per_sample_2024-08-12_12-26.csv")
# ACTIVATIONS_FILE = Path("../foldingnet2/snapshot/Reconstruct_insect_foldnet_gaussian_k20_e1600/features/activations_per_sample_2024-07-30_18-04.csv")
# ACTIVATIONS_FILE = Path("../foldingnet2/snapshot/Reconstruct_insect_foldnet_gaussian_k20_e2000_feat512_pts4096_augment/features/activations_per_sample_2024-08-03_22-16.csv")
# ACTIVATIONS_FILE = Path("../foldingnet2/snapshot/Reconstruct_insect_foldnet_gaussian_k40_e1600_feat1024_pts4096_augment_3/features/activations_per_sample_2024-08-09_16-49.csv")
# DATASET_DIR = Path("../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_3")
DATASET_DIR = Path("../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_4")
# DATASET_DIR = Path("../../datasets/insect/100ms_2048pts_fps-ds_sor-nr_norm_shufflet_1")
# DATASET_SPLIT_FILE = Path("../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_3/train_test_split_40shot_1.txt")
DATASET_SPLIT_FILE = Path("../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_4/train_test_split_40shot.txt")
# Contains exported 2d projections
FIGURES_DIR = Path("output/figures/projection_and_hist/tf100ms_tbr250_2024-08-07_23-58-41")
# Labels mapping file will be exported to this dir
LABELS_OUTPUT_DIR = Path("output/instance_classes/tsne_inspector")
# Dir with bbox annotations; Existing files will be overwritten!
ANNOTATIONS_DIR = Path("output/video_annotations/3_classified")
SHOW_TEST_AS_UNLABELED = False
GROUP_TO_TRAJECORIES = True # true: Points are whole trajectories; false: Points are fragments


def get_projected_heatmap(df, col1, col2, bins_x, bins_y):
    df_proj = df.loc[:,[col1, col2]]

    heatmap, xedges, yedges = np.histogram2d(df_proj[col1], df_proj[col2], bins=[bins_x, bins_y], density=False)
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



class LassoManager:
    def __init__(self, fig, collection, callback, useblit=True):
        self.fig = fig
        self.collection = collection
        self.callback = callback
        self.useblit = useblit
        self.lasso = None

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        canvas = self.fig.canvas
        if event.inaxes is not self.collection.axes or canvas.widgetlock.locked():
            return
        self.lasso = Lasso(event.inaxes, (event.xdata, event.ydata), self.callback, self.useblit)
        canvas.widgetlock(self.lasso)  # acquire a lock on the widget drawing

    def on_release(self, event):
        canvas = self.fig.canvas
        if hasattr(self, 'lasso') and canvas.widgetlock.isowner(self.lasso):
            canvas.widgetlock.release(self.lasso)


class TsneInspector:
    """ Inspect samples and assign labels interactively in a t-sne plot. """

    point_sizes = [10,25,50,100]

    def __init__(self, activations_file, dataset_dir, figures_dir, labels_output_dir, annotations_dir, split_file) -> None:
        self.activations_file = activations_file
        self.dataset_dir = dataset_dir
        self.figures_dir = figures_dir
        self.labels_output_dir = labels_output_dir
        self.annotations_dir = annotations_dir
        self.split_file = split_file
        self.split = "all"
        self.fig = None
        # selected with lasso or by click
        self.selected_ind = np.array([], dtype=int)
        self.point_size_index = 1
        self.show_test_as_unlabeled = SHOW_TEST_AS_UNLABELED

        self.create_dfs()
        self.create_tsne()
        self.filter_df()
        self.create_figure()
        self.create_ui()


    def create_dfs(self):
        # load activations file
        act_file_df = pd.read_csv(self.activations_file, sep=",", header="infer")

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

        if self.split_file is not None:
            # find out split ("train"/"test") of each fragment
            # fid example: "hn-dra-1_16_6"
            train_fids, test_fids = bee.read_split_file(DATASET_SPLIT_FILE)
            df["split"] = df["frag_id"].apply(lambda frag_tuple: self.apply_split(frag_tuple, train_fids, test_fids))
        else:
            df["split"] = ""

        if self.show_test_as_unlabeled:
            sel = df["split"]=="test"
            df.loc[sel,"orig_target_name"] = "insect"
            df.loc[sel,"target_name"] = "insect"
            df.loc[sel,"target_index"] = 1

        # activations / features
        # find range of activation columns: "act_0" - "act_N"
        self.max_act_col = [c for c in act_file_df.columns if str(c).startswith("act_")][-1]
        print("Using activation columns: act_0 -", self.max_act_col)
        # get all "act_X"-Columns
        act_df = act_file_df.loc[:,"act_0":self.max_act_col].copy()
        df = pd.concat([df, act_df], axis=1)

        # group fragments to trajecotries
        if GROUP_TO_TRAJECORIES:
            self.frags_df = df
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

        self.full_df = df

        # if self.split != "all":
        #     self.df = self.df.drop(self.df[self.df["split"] != self.split].index)
        #     self.df.reset_index(drop=True, inplace=True)


    def create_tsne(self):
        # Create 2D t-sne
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=0)
        act_df = self.full_df.loc[:,"act_0":self.max_act_col].copy()
        tsne_result = tsne.fit_transform(act_df)

        # Add x,y to dataframe
        tsne_df = pd.DataFrame(tsne_result, columns=["tsne_x", "tsne_y"])
        self.full_df = pd.concat([self.full_df, tsne_df], axis=1)

        # print(self.df.head())


    def create_figure(self):
        # Create plot
        subplot_kw = dict(xlim=(-10, 10), ylim=(-10, 10), autoscale_on=False)
        gridspec_kw = dict(height_ratios=[100,20,20])
        self.fig, (self.ax_tsne, self.ax_tx, self.ax_ty) = plt.subplots(3, 1, figsize=(6, 8), gridspec_kw=gridspec_kw)

        # array of tuple to 2d-array
        # self.fc = np.array([*(bee.get_rgba_of_class_index(self.descr_df["target_index"]).to_numpy())])

        point_size = TsneInspector.point_sizes[self.point_size_index]
        self.scatter = self.ax_tsne.scatter(x=self.get_df().loc[:,"tsne_x"], y=self.get_df().loc[:,"tsne_y"], \
                             s=point_size, picker=True, pickradius=10, linewidth=1.5) # c=self.df["target_index"], cmap="Set1", vmin=0, vmax=8, c=self.fc

        self.update_colors()
        
        self.ax_tsne.set_title("t-SNE plot of last inner layer activations")
        self.ax_tx.set_title("t-x-projection")
        self.ax_ty.set_title("t-y-projection")

        for ax in (self.ax_tsne, self.ax_tx, self.ax_ty):
            ax.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False) 

        # Events
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        # not needed right now
        # self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        # Lasso handler
        # self.lasso = LassoSelector(self.ax_tsne, onselect=self.on_lasso, useblit=False)
        self.lasso_manager = LassoManager(self.fig, self.scatter, self.on_lasso, useblit=True)

        # create legend
        legend_handles = []
        for i,cla in enumerate(bee.CLASSES):
            clr = bee.get_rgba_of_class_index(i, 0.66)
            h = mpatches.Patch(color=clr, label=cla)
            legend_handles.append(h)
        self.ax_tsne.legend(handles=legend_handles)

        self.fig.tight_layout()


    def create_ui(self):
        """  Create the whole UI. """

        if self.fig is None:
            self.create_plot()
        fig = self.get_figure()

        root = tk.Tk()
        self.root = root
        root.wm_title("t-SNE Sample Inspector")

        frame_left = tk.Frame(master=root)
        # frame_left.grid(row=0, column=0)
        frame_left.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

        canvas = FigureCanvasTkAgg(fig, master=frame_left)
        canvas.draw()
        canvas.mpl_connect("key_press_event", lambda event: print(f"you pressed {event.key}"))
        canvas.mpl_connect("key_press_event", key_press_handler)
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # pack_toolbar=False will make it easier to use a layout manager later on.
        toolbar = NavigationToolbar2Tk(canvas, frame_left, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)

        # Right frame
        frame_right = tk.Frame(master=root)
        # frame_right.grid(row=0, column=1, padx=10, pady=5, sticky="N")
        frame_right.pack(fill=tk.BOTH, side=tk.RIGHT, expand=True)

        # This label sets the width for the right frame!
        tk.Label(frame_right, text="Selected:", width=30, pady=5, anchor="w", font=("Consolas", 11) )\
            .pack(side=tk.TOP, fill=tk.X)

        self.scene_id_var = tk.StringVar()
        self.label_scene_id = tk.Label(frame_right, textvariable=self.scene_id_var, pady=0, anchor="w", font=("Consolas", 11) )
        self.label_scene_id.pack(side=tk.TOP, fill=tk.X)

        self.instance_id_var = tk.StringVar()
        self.label_instance_id = tk.Label(frame_right, textvariable=self.instance_id_var, pady=0, anchor="w", font=("Consolas", 11) )
        self.label_instance_id.pack(side=tk.TOP, fill=tk.X)

        self.fragment_index_var = tk.StringVar()
        self.label_fragment_index = tk.Label(frame_right, textvariable=self.fragment_index_var, pady=0, anchor="w", font=("Consolas", 11) )
        self.label_fragment_index.pack(side=tk.TOP, fill=tk.X)

        self.class_name_var = tk.StringVar()
        self.label_class = tk.Label(frame_right, textvariable=self.class_name_var, pady=0, anchor="w", font=("Consolas", 11) )
        self.label_class.pack(side=tk.TOP, fill=tk.X, pady=(0,10))

        self.update_selection_labels()

        tk.Label(frame_right, text="Assign class to selection:", width=30, pady=5, anchor="w", font=("Consolas", 11) )\
                .pack(side=tk.TOP, fill=tk.X)

        self.class_var = tk.IntVar(master=root, value=-1)
        self.rb_classes = []
        for class_index,class_name in enumerate(bee.CLASSES): #  + ["no class"]
            rb = tk.Radiobutton(frame_right, text=class_name.upper(), variable=self.class_var, value=class_index, anchor="w", \
                                command=lambda: self.rb_classes_value_change(rb), font=("Consolas", 10))
            rb.pack(side=tk.TOP, fill=tk.X)
            self.rb_classes.append(rb)


        tk.Label(frame_right, text="Visible dataset split:", width=30, pady=5, anchor="w", font=("Consolas", 11) )\
            .pack(side=tk.TOP, fill=tk.X)

        self.split_vals = ["all","train","test"]
        self.split_var = tk.StringVar(master=root, value=self.split_vals[0])
        self.rbs_split = []
        for i,split_val in enumerate(self.split_vals):
            rb = tk.Radiobutton(frame_right, text=split_val.upper(), variable=self.split_var, value=split_val, anchor="w", \
                                command=lambda: self.rbs_split_value_change(rb), font=("Consolas", 10))
            rb.pack(side=tk.TOP, fill=tk.X)
            self.rbs_split.append(rb)

        button_unselect = tk.Button(master=frame_right, text="Unselect all", width=20, command=self.unselect_all)
        button_unselect.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        button_select_instance = tk.Button(master=frame_right, text="Select all of instance", width=20, command=self.select_all_frags_of_selected_instance)
        button_select_instance.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        button_show_full_traj = tk.Button(master=frame_right, text="Show full trajectory", width=20, command=self.show_full_trajectory)
        button_show_full_traj.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        button_change_point_size = tk.Button(master=frame_right, text="Change point size ", width=20, command=self.change_point_size)
        button_change_point_size.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        # Save buttons
        button_save_labels_video_ann = tk.Button(master=frame_right, text="Save: Update annotations file ", width=20, \
                                                 command=self.save_labels_overwrite_video_annotations)
        button_save_labels_video_ann.pack(side=tk.BOTTOM, padx=10, pady=5, fill=tk.X)

        button_save_as_new_csv = tk.Button(master=frame_right, text="Save: As separate csv ", width=20, \
                                                 command=self.save_labels_as_new_csv)
        button_save_as_new_csv.pack(side=tk.BOTTOM, padx=10, pady=5, fill=tk.X)


    def show(self):
        tk.mainloop()

    def get_df(self):
        return self.df
        # if self.split == "train" or self.split == "test":
        #     return self.full_df[ self.full_df["split"] == self.split ].reset_index()
        # else:
        #     return self.full_df
    
    def filter_df(self):
        # pass
        if self.split == "train" or self.split == "test":
            self.df = self.full_df[self.full_df["split"] == self.split].reset_index().rename(columns={"index":"full_index"})
        else:
            self.df = self.full_df
        

    def update_figure_points(self):
        self.scatter.set_offsets(self.get_df().loc[:,["tsne_x","tsne_y"]])
        self.update_colors()
        self.fig.canvas.draw_idle()

    def get_figure(self):
        return self.fig

    def show_figure(self):
        self.fig.show()

    def unselect_all(self):
        self.select(np.array([], dtype=int))

    def select(self, selected_ind):
        prev_selected_ind = self.selected_ind
        self.selected_ind = selected_ind

        # How many points are selectd?
        if len(selected_ind) == 0:
            self.update_selection_labels()
        elif len(selected_ind) == 1:
            ind = selected_ind[0]
            frag_id = self.get_df().at[ind, "frag_id"]
            target_name = self.get_df().at[ind, "target_name"].upper()
            sample_path = self.get_df().at[ind, "sample_path"]
            # Update the UI
            self.update_selection_labels(frag_id[0], frag_id[1], frag_id[2], target_name)
            self.show_fragment(sample_path)
        else:
            self.update_selection_labels(fragment_index=f"multiple ({len(selected_ind)})")

        # Set border color to black
        self.ec[prev_selected_ind] = self.fc[prev_selected_ind]
        self.ec[selected_ind] = (0.0,0.0,0.0,1.0)
        self.scatter.set_edgecolor(self.ec)

        # Set border width
        # self.lw[prev_selected_ind] = 1.5
        # self.lw[selected_ind] = 2.0
        # self.scatter.set_linewidth(self.lw)

        self.update_class_radiobuttons(selected_ind)
        self.fig.canvas.draw_idle()

    def select_all_frags_of_selected_instance(self):
        """
        Select all fragments (points) of currently selected instance (or of multiple instances).
        """
        # Find selected ids
        selected_instances = self.get_df().loc[self.selected_ind, ["scene_id", "instance_id"]]
        self.select_all_frags_of_instance(selected_instances)

    def select_all_frags_of_instance(self, scene_and_instance_ids:pd.DataFrame):
        """
        Args:
            scene_and_instance_ids (pd.DataFrame): Must contain columns "scene_id", "instance_id"
        """
        frag_inds = self.get_all_frags_of_instance(scene_and_instance_ids)
        self.select(frag_inds)

    def get_all_frags_of_instance(self, scene_and_instance_ids:pd.DataFrame, return_list=True):
        """
        Find all fragments of the given instances.
        Args:
            scene_and_instance_ids (pd.DataFrame): _description_
            return_list (bool, optional): _description_. Defaults to True.
        Returns:
            list | pd.DataFrame: frag_inds
        """
        scene_and_instance_ids = scene_and_instance_ids.loc[:,["scene_id", "instance_id"]]
        scene_and_instance_ids = scene_and_instance_ids.drop_duplicates()
        # Find selected rows
        df = self.get_df()
        df["index"] = df.index
        merged = df.merge(scene_and_instance_ids, on=["scene_id", "instance_id"], how='left', indicator="merge").set_index('index')
        # frag inds will be a Series containing all rows with True or False
        frag_inds = merged["merge"]=="both"
        if return_list:
            # convert df of True/False values to list of indices
            # get list of indices of rows that are selected
            frag_inds = frag_inds[frag_inds]
            frag_inds = frag_inds.index[frag_inds == True].tolist()
        return frag_inds

    def update_selection_labels(self, scene_id=None, instance_id=None, fragment_index=None, target_name=None):
            self.scene_id_var.set(          f"Scene:     {scene_id}")
            self.instance_id_var.set(       f"Instance:  {instance_id}")
            self.fragment_index_var.set(    f"Part:      {fragment_index}")
            self.class_name_var.set(        f"Class:     {target_name}")

    def update_class_radiobuttons(self, selected_ind):
        if len(selected_ind) == 1:
            ind = selected_ind[0]
            class_index = self.get_df().at[ind, "target_index"]
            self.class_var.set(int(class_index))
        else:
            self.class_var.set(-1)


    def rb_classes_value_change(self, rb):
        if self.selected_ind is None or len(self.selected_ind)==0:
            print("No sample selected!")
            return
        class_index = self.class_var.get()
        self.set_class_of_samples(self.selected_ind, class_index)


    def rbs_split_value_change(self, rb):
        self.split = self.split_var.get() # "all", "train" or "test"
        self.filter_df()
        self.unselect_all()
        # Update plot
        self.update_figure_points()


    def set_class_of_samples(self, selected_ind, new_class_index):
        new_class_name = bee.CLASSES[new_class_index]
        
        # Find selected fragments
        selected_instances = self.get_df().loc[selected_ind, ["scene_id", "instance_id"]]
        frag_inds = self.get_all_frags_of_instance(selected_instances, return_list=False)

        # update target_index and target_name columns of all fragments
        self.get_df().loc[frag_inds, ["target_index","target_name"]] = new_class_index, new_class_name

        # TODO Move and rename fragment files to other class dir?
        # or do this in a separate function/button event?

        
        self.update_colors()
        self.fig.canvas.draw_idle()


    def update_colors(self):
        # Doesnt really work!!!
        # face colors
        # array of tuple to 2d-array
        self.fc = np.array([ *( bee.get_rgba_of_class_index(self.get_df().loc[:,"target_index"], 0.66).to_numpy() ) ])
        self.scatter.set_facecolor(self.fc)
        # edge colors
        self.ec = self.fc.copy()
        self.ec[:,3] = 1.0
        self.ec[self.selected_ind] = (0,0,0,1)
        self.scatter.set_edgecolor(self.ec)


    def on_pick(self, event):
        if event.artist != self.scatter:
            return
        if len(event.ind) == 0:
            return
        
        # find closest point to cursor
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata
        df = self.get_df()
        distances = np.hypot(x - df.loc[event.ind,"tsne_x"], y - df.loc[event.ind,"tsne_y"])
        indmin = distances.argmin()
        closest_index = event.ind[indmin]

        # select point
        self.select(np.array([closest_index]))


    def on_lasso(self, verts):
        if len(verts) < 30:
            # not enough verts
            self.fig.canvas.draw_idle()
            return
        path = matplotlib.path.Path(verts)
        # drop previous selection:
        pts = self.scatter.get_offsets()
        selected_ind = np.nonzero(path.contains_points(pts))[0].astype(int)
        self.select(selected_ind)


    # def on_move(self, event):
    #     """ not used! """
    #     if event.inaxes != self.ax_tsne:
    #         return
    #     # print(f'data coords {event.xdata} {event.ydata},', f'pixel coords {event.x} {event.y}')
    #     x = event.xdata
    #     y = event.ydata
    #     distances = np.hypot(x - self.tsne_result[:,0], y - self.tsne_result[:,1])
    #     closest_ind = distances.argmin()
    #     frag_id = self.df.at[closest_ind, "frag_id"]
    #     target_name = self.df.at[closest_ind, "target_name"]
    #     sample_path = self.df.at[closest_ind, "sample_path"]
    #     # Update the UI
    #     self.selected_id_var.set(f"Closest sample: {frag_id}")


    def show_fragment(self, sample_path):
        frag_path = self.dataset_dir / sample_path
        if not frag_path.is_file():
            print("WARNING: Fragment " + str(frag_path) + " does not exist!")
            return
        
        self.ax_tx.clear()
        self.ax_ty.clear()

        frag_df = pd.read_csv(frag_path, sep=",", header="infer")

        # Show projection as 2D histogram
        # tx_df, tx_heatmap, tx_extent = get_projected_heatmap(frag_df, "t", "x", 500, 40)
        # ty_df, ty_heatmap, ty_extent = get_projected_heatmap(frag_df, "t", "y", 500, 40)
        
        # tx_heatmap_image = hist2d_to_image(tx_heatmap)
        # ty_heatmap_image = hist2d_to_image(ty_heatmap)
        
        # self.ax_tx.imshow(tx_heatmap_image, origin='upper', cmap="gray", aspect="auto")
        # self.ax_ty.imshow(ty_heatmap_image, origin='upper', cmap="gray", aspect="auto")

        # alternative: Show projection as scatter plot
        self.ax_tx.scatter(x=frag_df["t"], y=frag_df["x"], c="black", s=0.5, alpha=0.08)
        self.ax_ty.scatter(x=frag_df["t"], y=frag_df["y"], c="black", s=0.5, alpha=0.08)

        self.ax_tx.set_title("t-x-projection")
        self.ax_ty.set_title("t-y-projection")

        # self.fc[:, -1] = 0.2
        # self.scatter.set_facecolors(self.fc)

        self.fig.canvas.draw_idle()


    def show_full_trajectory(self):
        if len(self.selected_ind)==0:
            print("ERROR: No sample selected!")
        elif len(self.selected_ind)==1:
            ind = self.selected_ind[0]
            scene_id, instance_id, frag_index = self.get_df().at[ind, "frag_id"]
            scene_name = bee.scene_aliases_by_id(scene_id)[0]
            scene_dir = self.figures_dir / (scene_name+"_trajectories")

            if not scene_dir.exists():
                print("ERROR: Directory " + str(scene_dir) + " does not exist!")
                return

            # print("Searching path", (scene_dir/"txproj"))
            found_paths = list((scene_dir/"txproj").glob(f"{instance_id}_*_txproj.png"))
            if len(found_paths)==0:
                print("ERROR: No 2D projection found at", (scene_dir/"txproj"))
            elif len(found_paths)>1:
                print("ERROR: Too many 2D projections found at", (scene_dir/"txproj"))
            else: # ==1
                # load tx and ty projections
                tx_proj_path = found_paths[0]
                ty_proj_path = scene_dir / "typroj" / tx_proj_path.name.replace("_txproj", "_typroj")
                # print(tx_proj_path)
                # print(ty_proj_path)
                tx_proj = plt.imread(tx_proj_path)
                ty_proj = plt.imread(ty_proj_path)
                # show
                fig1, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
                ax1.imshow(tx_proj, origin='upper', cmap="gray") # , aspect="auto" -> verzerrt das Bild; Auch bei zoom!
                ax2.imshow(ty_proj, origin='upper', cmap="gray")
                ax1.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False) 
                ax2.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False) 
                fig1.canvas.manager.set_window_title(f't-y- and t-y-projection of scene: {scene_id}, instance: {instance_id}, fragment: {frag_index}')
                fig1.tight_layout()
                fig1.show()

        else:
            print("ERROR: Too many samples selected! Select only one!")


    def save_labels_overwrite_video_annotations(self):
        """ Update annotations file in output/video_annotations/3_classified/ """
        # Do backup of file first.
        # On start of program print warning if assigned classes from video_annotations file and dataset-classes dont match.

        # Problem: Class assignments are in multiple files: activations file, annotations file and in dataset dir structure!

        # scene_ids = self.descr_df["scene_id"].unique()
        # print(scene_ids)
        # for scene_id in scene_ids:
        #     print(bee.scene_aliases_by_id(scene_id)[0])
        #     # load csv
        #     scene_name = bee.scene_aliases_by_id(scene_id)[0]
        #     ann_file = self.annotations_dir / (scene_name+".csv")
        #     if not ann_file.exists():
        #         print("ERROR: Cannot update annotations file; File does not exist;", ann_file)
        #         continue
        #     columns = ["frame_index", "class", "instance_id", "is_difficult", "x","y","w","h"]
        #     df = pd.read_csv(ann_file, sep=",", header=0, names=columns)
        #     print(df)
        #     break
        print("NOT IMPLEMENTED!")


    def save_labels_as_new_csv(self):
        """ Create new csv file with id-class mappings. Save at output/instance_classes/instance_classes/. """

        df = self.get_df()

        # Check if all intances have exactly one class.
        # If there are fragments of the same instance with different classes this will return false
        all_have_one_class = (df.groupby(["scene_id", "instance_id"]).nunique()["target_name"] == 1).all()
        if not all_have_one_class:
            print("ERROR: Cannot save labels; Some instances have ambiguous (more than one) labels!")


        df = df.groupby(["scene_id", "instance_id"]).first()[["target_name", "orig_target_name"]]
        changed_inds = df["target_name"] != df["orig_target_name"]
        number_of_changed_instances = changed_inds.sum()

        print("Changes:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(df.loc[changed_inds, ["orig_target_name","target_name"]])

        # output columns: scene_id, instance_id, class
        output_df = pd.DataFrame()
        output_df["class"] = df["target_name"]

        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = LABELS_OUTPUT_DIR / f"instance_classes_{datetime_str}.csv"
        output_file.parent.mkdir(exist_ok=True, parents=True)
        output_df.to_csv(output_file, sep=",", header=True, index=True)
        print(f"Saved as {output_file}; {number_of_changed_instances} instance labels were changed!")


    def change_point_size(self):
        self.point_size_index = (self.point_size_index + 1) % len(TsneInspector.point_sizes)
        point_size = TsneInspector.point_sizes[self.point_size_index]
        self.scatter.set_sizes([point_size])
        self.fig.canvas.draw_idle()

    def apply_split(self, frag_tuple, train_fids, test_fids):
        fid = bee.id_tuple_to_str(frag_tuple)
        if fid in train_fids:
            return "train"
        elif fid in test_fids:
            return "test"
        else:
            return None
        


if __name__ == '__main__':
    tsni = TsneInspector(ACTIVATIONS_FILE, DATASET_DIR, FIGURES_DIR, LABELS_OUTPUT_DIR, \
                         ANNOTATIONS_DIR, DATASET_SPLIT_FILE)
    tsni.show()
    

        
    
