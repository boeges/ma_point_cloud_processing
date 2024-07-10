
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

# ACTIVATIONS_FILE = Path("../Pointnet_Pointnet2_pytorch/log/classification/2024-07-03_23-11/logs/activations_per_class_2024-07-07_21-31.csv")
ACTIVATIONS_FILE = Path("../Pointnet_Pointnet2_pytorch/log/classification/2024-07-03_23-11/logs/activations_per_class_2024-07-09_23-46.csv")
# DATASET_DIR = Path("../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_2024-07-03_23-04-52")
DATASET_DIR = Path("../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_2024-07-09_22-50-18")
# Contains exported 2d projections
FIGURES_DIR = Path("output/figures/projection_and_hist/tf100ms_tbr250_pred_2024-07-06")


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

    def __init__(self, activations_file, dataset_dir, figures_dir) -> None:
        self.activations_file = activations_file
        self.dataset_dir = dataset_dir
        self.figures_dir = figures_dir
        self.fig = None
        # selected with lasso or by click
        self.selected_ind = np.array([], dtype=int)

        self.create_dfs()
        self.create_tsne()
        self.create_figure()
        self.create_ui()


    def create_dfs(self):
        # load activations file
        df = pd.read_csv(self.activations_file, sep=",", header="infer")

        # Split df up in description df and activations df
        # description df
        self.descr_df = df[["sample_path", "target_name"]].copy()
        # add "frag_id" column
        self.descr_df.loc[:,"frag_id"] = self.descr_df.loc[:,'sample_path'].apply(bee.frag_filename_to_id)
        # add "target_index" column
        classes_map = dict(zip(bee.CLASSES, range(len(bee.CLASSES))))
        self.descr_df.loc[:,'target_index'] = self.descr_df.loc[:,'target_name'].map(classes_map)

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

        self.scatter = self.ax_tsne.scatter(x=self.tsne_result[:,0], y=self.tsne_result[:,1], \
                             s=20, picker=True, pickradius=5) # c=self.df["target_index"], cmap="Set1", vmin=0, vmax=8, c=self.fc

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
        self.selected_id_var = tk.StringVar(value="Selected: no sample selected")
        self.label_id = tk.Label(frame_right, textvariable=self.selected_id_var, width=30, pady=10, anchor="w", font=("Arial", 11) )
        self.label_id.pack(side=tk.TOP, fill=tk.X)

        self.class_var = tk.IntVar(master=root, value=-1)
        self.rb_classes = []
        for class_index,class_name in enumerate(bee.CLASSES + ["no class"]):
            rb = tk.Radiobutton(frame_right, text=class_name, variable=self.class_var, value=class_index, anchor="w", \
                                command=lambda: self.rb_value_change(rb))
            rb.pack(side=tk.TOP, fill=tk.X)
            self.rb_classes.append(rb)

        button_unselect = tk.Button(master=frame_right, text="Unselect all", width=20, command=self.unselect_all)
        button_unselect.pack(side=tk.TOP, pady=5)

        button_show_full_traj = tk.Button(master=frame_right, text="Show full trajectory", width=20, command=self.show_full_trajectory)
        button_show_full_traj.pack(side=tk.TOP, pady=5)


    def show(self):
        tk.mainloop()

    def get_figure(self):
        return self.fig

    def show_figure(self):
        self.fig.show()

    def unselect_all(self):
        self.select(np.array([], dtype=int))

    def select(self, selected_ind):
        self.selected_ind = selected_ind

        # set border color
        self.ec[:] = self.fc[:]
        self.ec[selected_ind] = (0,0,0,1)
        self.scatter.set_edgecolor(self.ec)

        # Set border width
        # self.lw[closest_ind] = 1.0
        # self.scatter.set_linewidth(self.lw)

        # How many points are selectd?
        if len(selected_ind) == 0:
            self.selected_id_var.set("Selected: no sample selected")
        elif len(selected_ind) == 1:
            ind = selected_ind[0]
            frag_id = self.descr_df.at[ind, "frag_id"]
            target_name = self.descr_df.at[ind, "target_name"]
            sample_path = self.descr_df.at[ind, "sample_path"]
            # Update the UI
            self.selected_id_var.set(f"Selected: {frag_id} ({target_name})")
            self.show_fragment(sample_path)
        else:
            self.selected_id_var.set(f"Selected: multiple ({len(selected_ind)})")

        self.update_class_radiobuttons(selected_ind)
        self.fig.canvas.draw_idle()


    def update_class_radiobuttons(self, selected_ind):
        if len(selected_ind) == 1:
            ind = selected_ind[0]
            class_index = self.descr_df.at[ind, "target_index"]
            self.class_var.set(int(class_index))
            # for rb in self.rb_classes:
            #     rb.configure(state = tk.NORMAL)
        else:
            self.class_var.set(-1)
            # for rb in self.rb_classes:
            #     rb.configure(state = tk.DISABLED)


    def rb_value_change(self, rb):
        class_index = self.class_var.get()
        self.set_class_of_samples(self.selected_ind, class_index)


    def set_class_of_samples(self, selected_ind, class_index):
        if class_index==-1 or class_index==len(bee.CLASSES):
            class_index = -1
            class_name = None
        else:
            class_name = bee.CLASSES[class_index]
        # set new values
        self.descr_df.loc[selected_ind, "target_index"] = class_index
        self.descr_df.loc[selected_ind, "target_name"] = class_name

        self.update_colors()

        self.fig.canvas.draw_idle()


    def update_colors(self):
        # face colors
        # array of tuple to 2d-array
        self.fc = np.array([*(bee.get_rgba_of_class_index(self.descr_df["target_index"]).to_numpy())])
        self.scatter.set_facecolor(self.fc)
        # edge colors
        self.ec = self.fc.copy()
        self.ec[self.selected_ind] = (0,0,0,1)
        self.scatter.set_edgecolor(self.ec)


    def on_pick(self, event):
        if event.artist != self.scatter:
            return

        # event.ind returns all point indices in radius
        N = len(event.ind)
        if not N:
            return True
        
        # clear previous selection
        if self.selected_ind is not None:
            self.ec[self.selected_ind] = self.fc[self.selected_ind]
            # self.lw[self.selected_ind] = 1.0

        # find closest point to cursor
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - self.tsne_result[:,0][event.ind], y - self.tsne_result[:,1][event.ind])
        indmin = distances.argmin()
        closest_index = event.ind[indmin]
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
            scene_name = bee.scene_aliases_by_short_id(scene_id)[0]
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
                # load image
                img_path = found_paths[0]
                img = plt.imread(img_path)
                # show
                fig1, ax1 = plt.subplots(figsize=(12, 8))
                ax1.imshow(img, origin='upper', cmap="gray") # aspect="auto"
                ax1.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False) 
                fig1.tight_layout()
                fig1.show()

        else:
            print("ERROR: Too many samples selected! Select only one!")





if __name__ == '__main__':
    tsni = TsneInspector(ACTIVATIONS_FILE, DATASET_DIR, FIGURES_DIR)
    tsni.show()
    

        
    
