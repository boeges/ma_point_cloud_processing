# use conda env "pix2pix"

from pathlib import Path
from datetime import datetime

import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

import bee_utils as bee


DATASET_DIR = Path("../../datasets/insect/100ms_4096pts_rnd-ds_sor-nr_norm_shufflet_5")
OUTPUT_DIR = Path("output/fragment_labels/")
STARTING_INDEX = 0
STARTING_SAMPLE_ID = "hn-dra-1_10_4"
# LOAD_FILE = Path("output/fragment_labels/fragment_easy_difficult_2024-09-01_23-54-11.csv")
LOAD_FILE = None


class FragmentLabeller:
    def __init__(self):
        self.dataset_dir = DATASET_DIR
        self.labels_output_dir = OUTPUT_DIR
        self.loaded_index = STARTING_INDEX
        self.starting_sample_id = STARTING_SAMPLE_ID

        load_file = None
        if LOAD_FILE is not None and LOAD_FILE.exists():
            load_file = LOAD_FILE

        self.load_dataset(load_file)
        self.create_figure()
        self.create_ui()
        self.load_sample_by_id(self.starting_sample_id)
        


    def load_dataset(self, load_file=None):
        sample_paths_and_ids = [[(p.parent.name+"/"+p.name), p.stem] for p in self.dataset_dir.glob("*/*.csv")]
        paths = [p[0] for p in sample_paths_and_ids]
        ids = [p[1] for p in sample_paths_and_ids]
        df = pd.DataFrame(data={"sample_path":paths, "sample_id": ids})
        df["scene_id"] = df["sample_id"].apply(lambda v: v.split("_")[0])
        df["instance_id"] = df["sample_id"].apply(lambda v: v.split("_")[1])
        df["fragment_index"] = df["sample_id"].apply(lambda v: v.split("_")[2])

        df["is_difficult"] = None
        # if load_file is None:
        #     df["is_difficult"] = None
        # else:
        #     print("Loading existing file from", load_file)
        #     df1 = pd.read_csv(load_file, header="infer")
        #     print("is_difficult unique values", df1["is_difficult"].unique())
        #     df1["is_difficult"].fillna(value="", inplace=True)
        #     df1["is_difficult"] = df1["is_difficult"].round().astype(int).astype(str)
        #     print("File contains x annotations", len(df1[df1["is_difficult"] != ""].index))
        #     df1 = df1[["sample_id","is_difficult"]]
        #     df = pd.merge(df, df1, left_on="sample_id", right_on="sample_id", how="left", validate="one_to_one")
        #     print("is_difficult unique values", df["is_difficult"].unique())

        self.df = df

        print(df)


    def create_figure(self):
        # Create plot
        self.fig, (self.ax_tx, self.ax_ty) = plt.subplots(2, 1, figsize=(6, 4))
        
        self.ax_tx.set_title("t-x-projection")
        self.ax_ty.set_title("t-y-projection")

        for ax in (self.ax_tx, self.ax_ty):
            ax.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False) 

        self.fig.tight_layout()


    def create_ui(self):
        """  Create the whole UI. """

        root = tk.Tk()
        self.root = root
        root.wm_title("Fragment Labeller")

        frame_left = tk.Frame(master=root)
        # frame_left.grid(row=0, column=0)
        frame_left.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

        canvas = FigureCanvasTkAgg(self.fig, master=frame_left)
        canvas.draw()
        # canvas.mpl_connect("key_press_event", lambda event: print(f"you pressed {event.key}"))
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

        self.class_name_var = tk.StringVar()
        self.label_class = tk.Label(frame_right, textvariable=self.class_name_var, pady=0, anchor="w", font=("Consolas", 18) )
        self.label_class.pack(side=tk.TOP, pady=(0,5))

        self.sample_path_var = tk.StringVar()
        self.label_sample_path = tk.Label(frame_right, textvariable=self.sample_path_var, pady=0, anchor="w", font=("Consolas", 11) )
        self.label_sample_path.pack(side=tk.TOP)

        button_is_easy = tk.Button(master=frame_right, text="EASY", width=40, height=6, font=("Consolas", 16), \
                                   fg='black', bg='#99FF99', command=self.is_easy)
        button_is_easy.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        # button_is_medium = tk.Button(master=frame_right, text="MEDIUM", width=25, height=4, command=self.is_medium)
        # button_is_medium.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        button_is_difficult = tk.Button(master=frame_right, text="DIFFICULT", width=40, height=6, font=("Consolas", 16), \
                                        fg='black', bg='#FF9999', command=self.is_difficult)
        button_is_difficult.pack(side=tk.TOP, padx=10, pady=(5,20), fill=tk.X)

        button_save = tk.Button(master=frame_right, text="Save", width=20, command=self.on_save)
        button_save.pack(side=tk.BOTTOM, padx=10, pady=5, fill=tk.X)

        button_prev = tk.Button(master=frame_right, text="<-- go to previous               ", width=20, command=self.go_to_prev)
        button_prev.pack(side=tk.BOTTOM, padx=10, pady=5, fill=tk.X)

        button_next = tk.Button(master=frame_right, text="              go to next -->", width=20, command=self.go_to_next)
        button_next.pack(side=tk.BOTTOM, padx=10, pady=5, fill=tk.X)


    def show(self):
        tk.mainloop()

    def load_sample_by_id(self, sample_id):
        if sample_id is None:
            print("Loading sample at index 0")
            self.load_sample_by_index(0)
            return
        
        # find the index
        df = self.df
        df1 = df[df.sample_id == sample_id]
        print(df1)
        if len(df1.index) == 1:
            index = df1.index[0]
            print(f"Loading sample by name at index {index}")
            self.load_sample_by_index(index)
        else:
            raise RuntimeError("Failed to load sample by name!")
        

    def load_sample_by_index(self, index:int):
        df = self.df
        self.loaded_index = index

        dfi = df.loc[index,:]
        sample_path = dfi["sample_path"]
        full_path = DATASET_DIR / sample_path
        id = dfi["sample_id"]
        is_difficult = dfi["is_difficult"]
        is_diff_str = "---"
        if is_difficult == 1:
            is_diff_str = "difficult"
        if is_difficult == 0:
            is_diff_str = "easy"

        clas = sample_path.split("/")[0]
        self.class_name_var.set(clas)

        self.sample_path_var.set(id + f" / {is_diff_str}")

        frag_df = pd.read_csv(full_path, sep=",", header="infer")

        self.ax_tx.clear()
        self.ax_ty.clear()

        # Show projection as 2D histogram
        # tx_df, tx_heatmap, tx_extent = get_projected_heatmap(frag_df, "t", "x", 400, 40)
        # ty_df, ty_heatmap, ty_extent = get_projected_heatmap(frag_df, "t", "y", 400, 40)
        
        # tx_heatmap_image = hist2d_to_image(tx_heatmap)
        # ty_heatmap_image = hist2d_to_image(ty_heatmap)
        
        # self.ax_tx.imshow(tx_heatmap_image, origin='upper', cmap="gray", aspect="auto")
        # self.ax_ty.imshow(ty_heatmap_image, origin='upper', cmap="gray", aspect="auto")

        # alternative: Show projection as scatter plot
        self.ax_tx.scatter(x=frag_df["t"], y=frag_df["x"], c="black", s=1.5, alpha=0.1)
        self.ax_ty.scatter(x=frag_df["t"], y=frag_df["y"], c="black", s=1.5, alpha=0.1)

        self.ax_tx.set_title("t-x-projection")
        self.ax_ty.set_title("t-y-projection")

        self.ax_tx.invert_yaxis()
        self.ax_ty.invert_yaxis()

        self.fig.canvas.draw_idle()




    def go_to_next(self):
        self.load_sample_by_index(self.loaded_index+1)

    def go_to_prev(self):
        self.load_sample_by_index(self.loaded_index-1)


    def is_easy(self):
        self.df.loc[self.loaded_index,"is_difficult"] = "1"
        self.go_to_next()


    # def is_medium(self):
    #     self.df.loc[self.loaded_index,"is_difficult"] = False
    #     self.load_sample(self.loaded_index+1)


    def is_difficult(self):
        self.df.loc[self.loaded_index,"is_difficult"] = "0" # FIXME switch 0 and 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.go_to_next()


    def on_save(self):
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = OUTPUT_DIR / f"fragment_easy_difficult_{datetime_str}.csv"
        output_file.parent.mkdir(exist_ok=True, parents=True)
        self.df[["sample_id","is_difficult"]].to_csv(output_file, sep=",", header=True, index=False)
        print(f"Saved as {output_file}")




if __name__ == '__main__':
    labeller = FragmentLabeller()
    labeller.show()