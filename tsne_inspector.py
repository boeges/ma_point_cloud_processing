
# use conda env "pix2pix"

from pathlib import Path
from datetime import datetime

import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Lasso
from sklearn.manifold import TSNE

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

import bee_utils as bee


T_SCALE = 0.002
T_BUCKET_LENGTH = 1000 * 100 # 100ms

ACTIVATIONS_FILE = Path("../Pointnet_Pointnet2_pytorch/log/classification/2024-07-03_23-11/logs/activations_per_class_2024-07-23_12-30_with_bum.csv")
DATASET_DIR = Path("../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_2024-07-23_12-17-56")
# Contains exported 2d projections
FIGURES_DIR = Path("output/figures/projection_and_hist/tf100ms_tbr250_pred_2024-07-06")
# Labels mapping file will be exported to this dir
LABELS_OUTPUT_DIR = Path("output/instance_classes/tsne_inspector")
# Dir with bbox annotations; Existing files will be overwritten!
ANNOTATIONS_DIR = Path("output/video_annotations/3_classified")


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

    def __init__(self, activations_file, dataset_dir, figures_dir, labels_output_dir, annotations_dir) -> None:
        self.activations_file = activations_file
        self.dataset_dir = dataset_dir
        self.figures_dir = figures_dir
        self.labels_output_dir = labels_output_dir
        self.annotations_dir = annotations_dir
        self.fig = None
        # selected with lasso or by click
        self.selected_ind = np.array([], dtype=int)
        self.point_size_index = 1

        self.create_dfs()
        self.create_tsne()
        self.create_figure()
        self.create_ui()


    def create_dfs(self):
        # load activations file
        df = pd.read_csv(self.activations_file, sep=",", header="infer")

        # Split df up in description df and activations df
        # description df:
        # Columns: sample_path, targex_index, target_name, orig_target_name, frag_id:tuple, scene_id, instance_id, fragment_index
        self.descr_df = df[["sample_path", "target_name"]].copy()
        # self.descr_df.loc[:,"sample_path"].apply()
        self.descr_df["orig_target_name"] = self.descr_df["target_name"].copy()
        # add "frag_id" column
        self.descr_df.loc[:,"frag_id"] = self.descr_df.loc[:,'sample_path'].apply(bee.frag_filename_to_id)
        # split up frag_id-tuple into 3 columns
        self.descr_df[['scene_id', 'instance_id', "fragment_index"]] = pd.DataFrame(self.descr_df['frag_id'].tolist(), index=self.descr_df.index)
        # self.descr_df["scene_id"] = self.descr_df["scene_id"].apply(lambda s: bee.scene_aliases_by_short_id(s)[1])
        # add "target_index" column
        classes_map = dict(zip(bee.CLASSES, range(len(bee.CLASSES))))
        self.descr_df.loc[:,"target_index"] = self.descr_df.loc[:,"target_name"].map(classes_map)

        # activations df
        self.activations_df = df.loc[:,"act_0":"act_255"].copy()
        # print("df shape:", self.df.shape, "descr_df:", self.descr_df.shape, "activations_df:", self.activations_df.shape)


    def create_tsne(self):
        # Create 2D t-sne
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=0)
        self.tsne_result = tsne.fit_transform(self.activations_df)
        # print("tsne result shape", self.tsne_result.shape)


    def create_figure(self):
        # Create plot
        subplot_kw = dict(xlim=(-10, 10), ylim=(-10, 10), autoscale_on=False)
        gridspec_kw = dict(height_ratios=[100,20,20])
        self.fig, (self.ax_tsne, self.ax_tx, self.ax_ty) = plt.subplots(3, 1, figsize=(6, 8), gridspec_kw=gridspec_kw)

        # array of tuple to 2d-array
        # self.fc = np.array([*(bee.get_rgba_of_class_index(self.descr_df["target_index"]).to_numpy())])

        point_size = TsneInspector.point_sizes[self.point_size_index]
        self.scatter = self.ax_tsne.scatter(x=self.tsne_result[:,0], y=self.tsne_result[:,1], \
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
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # pack_toolbar=False will make it easier to use a layout manager later on.
        toolbar = NavigationToolbar2Tk(canvas, frame_left, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Right frame
        frame_right = tk.Frame(master=root)
        # frame_right.grid(row=0, column=1, padx=10, pady=5, sticky="N")
        frame_right.pack(fill=tk.BOTH, side=tk.RIGHT, expand=True)

        # This label sets the width for the right frame!
        self.label_selected_cap = tk.Label(frame_right, text="Selected:", width=30, pady=5, anchor="w", font=("Consolas", 11) )
        self.label_selected_cap.pack(side=tk.TOP, fill=tk.X)

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

        self.class_var = tk.IntVar(master=root, value=-1)
        self.rb_classes = []
        for class_index,class_name in enumerate(bee.CLASSES): #  + ["no class"]
            rb = tk.Radiobutton(frame_right, text=class_name.upper(), variable=self.class_var, value=class_index, anchor="w", \
                                command=lambda: self.rb_value_change(rb), font=("Consolas", 10))
            rb.pack(side=tk.TOP, fill=tk.X)
            self.rb_classes.append(rb)

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

    def get_figure(self):
        return self.fig

    def show_figure(self):
        self.fig.show()

    def unselect_all(self):
        self.select(np.array([], dtype=int))

    def select(self, selected_ind):
        prev_selected_ind = self.selected_ind
        self.selected_ind = selected_ind

        # set border color to black
        self.ec[prev_selected_ind] = self.fc[prev_selected_ind]
        self.ec[selected_ind] = (0.0,0.0,0.0,1.0)
        self.scatter.set_edgecolor(self.ec)

        # Set border width
        # self.lw[prev_selected_ind] = 1.5
        # self.lw[selected_ind] = 2.0
        # self.scatter.set_linewidth(self.lw)

        # How many points are selectd?
        if len(selected_ind) == 0:
            self.update_selection_labels()
        elif len(selected_ind) == 1:
            ind = selected_ind[0]
            frag_id = self.descr_df.at[ind, "frag_id"]
            target_name = self.descr_df.at[ind, "target_name"].upper()
            sample_path = self.descr_df.at[ind, "sample_path"]
            # Update the UI
            self.update_selection_labels(frag_id[0], frag_id[1], frag_id[2], target_name)
            self.show_fragment(sample_path)
        else:
            self.update_selection_labels(fragment_index=f"multiple ({len(selected_ind)})")

        self.update_class_radiobuttons(selected_ind)
        self.fig.canvas.draw_idle()

    def select_all_frags_of_selected_instance(self):
        """
        Select all fragments (points) of currently selected instance (or of multiple instances).
        """
        # Find selected ids
        selected_instances = self.descr_df.loc[self.selected_ind, ["scene_id", "instance_id"]]
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
        merged = self.descr_df.reset_index().merge(scene_and_instance_ids, on=["scene_id", "instance_id"], how='left', indicator="merge").set_index('index')
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
            class_index = self.descr_df.at[ind, "target_index"]
            self.class_var.set(int(class_index))
        else:
            self.class_var.set(-1)


    def rb_value_change(self, rb):
        if self.selected_ind is None or len(self.selected_ind)==0:
            print("No sample selected!")
            return
        class_index = self.class_var.get()
        self.set_class_of_samples(self.selected_ind, class_index)


    def set_class_of_samples(self, selected_ind, new_class_index):
        new_class_name = bee.CLASSES[new_class_index]
        
        # Find selected fragments
        selected_instances = self.descr_df.loc[selected_ind, ["scene_id", "instance_id"]]
        frag_inds = self.get_all_frags_of_instance(selected_instances, return_list=False)

        # update target_index and target_name columns of all fragments
        self.descr_df.loc[frag_inds, ["target_index","target_name"]] = new_class_index, new_class_name

        # TODO Move and rename fragment files to other class dir?
        # or do this in a separate function/button event?





        
        self.update_colors()
        self.fig.canvas.draw_idle()


    def update_colors(self):
        # Doesnt really work!!!
        # face colors
        # array of tuple to 2d-array
        self.fc = np.array([ *( bee.get_rgba_of_class_index(self.descr_df.loc[:,"target_index"], 0.66).to_numpy() ) ])
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
        distances = np.hypot(x - self.tsne_result[:,0][event.ind], y - self.tsne_result[:,1][event.ind])
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
        # to concat with previous selection:
        # self.selected_ind = np.unique(np.concatenate((self.selected_ind, ind), 0)).astype(int)
        # self.fc[:, -1] = 0.5
        # self.fc[self.selected_ind, -1] = 1
        # self.scatter.set_facecolors(self.fc)
        # self.fig.canvas.draw_idle()


    def on_move(self, event):
        """ not used! """

        if event.inaxes != self.ax_tsne:
            return
        
        # print(f'data coords {event.xdata} {event.ydata},', f'pixel coords {event.x} {event.y}')

        x = event.xdata
        y = event.ydata

        distances = np.hypot(x - self.tsne_result[:,0], y - self.tsne_result[:,1])
        closest_ind = distances.argmin()

        frag_id = self.descr_df.at[closest_ind, "frag_id"]
        target_name = self.descr_df.at[closest_ind, "target_name"]
        sample_path = self.descr_df.at[closest_ind, "sample_path"]

        # Update the UI
        self.selected_id_var.set(f"Closest sample: {frag_id}")


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
            scene_id, instance_id, frag_index = self.descr_df.at[ind, "frag_id"]
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

        # Check if all intances have exactly one class.
        # If there are fragments of the same instance with different classes this will return false
        all_have_one_class = (self.descr_df.groupby(["scene_id", "instance_id"]).nunique()["target_name"] == 1).all()
        if not all_have_one_class:
            print("ERROR: Cannot save labels; Some instances have ambiguous (more than one) labels!")


        df = self.descr_df.groupby(["scene_id", "instance_id"]).first()[["target_name", "orig_target_name"]]
        number_of_changed_instances = (df["target_name"] != df["orig_target_name"]).sum()

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


if __name__ == '__main__':
    tsni = TsneInspector(ACTIVATIONS_FILE, DATASET_DIR, FIGURES_DIR, LABELS_OUTPUT_DIR, ANNOTATIONS_DIR)
    tsni.show()
    

        
    
