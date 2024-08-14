import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    # 'font.family': 'serif',
    # 'font.size' : 11,
    # 'text.usetex': True,
    # 'pgf.rcfonts': False,
})

# np.random.seed(19680801)

# # example data
# mu = 100  # mean of distribution
# sigma = 15  # standard deviation of distribution
# x = mu + sigma * np.random.randn(437)

# num_bins = 50

# fig, ax = plt.subplots()

# # the histogram of the data
# n, bins, patches = ax.hist(x, num_bins, density=1)

# # add a 'best fit' line
# y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
# ax.plot(bins, y, '--')
# ax.set_xlabel('Smarts')
# ax.set_ylabel('Probability density')
# ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# # Tweak spacing to prevent clipping of ylabel
# fig.tight_layout()
# fig.set_size_inches(4.7747,3.5)
# plt.savefig('histogram.pgf')


TRAJECTORIES_CSV_DIR = Path("output/extracted_trajectories")

trajectory_filepath = TRAJECTORIES_CSV_DIR / \
    "3_classified/_with_bboxes/hauptsächlichBienen1_trajectories_bbox/fragments_time_100ms_4096pts_2024-06-10_23-45-35/9_bee_pts20400_start5289327/frag_9.csv"
# trajectory_filepath = TRAJECTORIES_CSV_DIR / \
#     "3_classified_pf1/_with_bboxes/hauptsächlichBienen1_trajectories_bbox/fragments_time_4000ms_4096pts_2024-06-12_15-10-31/18_bee_pts34743_start10928909/frag_1.csv"
# trajectory_filepath = TRAJECTORIES_CSV_DIR / \
#     "2_separated_2024-06-09_14-46-59/_with_bboxes/1_l-l-l_trajectories_bbox/fragments_time_100ms_4096pts_2024-06-10_23-17-48/0_ins_pts1073898_start1476947/frag_90.csv"

df = pd.read_csv(trajectory_filepath, sep=',', header="infer")
print(df.head())

events_df = df.loc[df["bb_corner_index"].astype(int) == -1]
bb_events_df = df[df["bb_corner_index"].astype(int) >= 0]

print(len(events_df), len(events_df.index), len(bb_events_df))


bboxes = {} # frame, bbox points
for i,row in bb_events_df.iterrows():
    x = int(row["x"])
    y = int(row["y"])
    t = float(row["t"])
    fi = int(row["bb_frame_index"])
    corner = int(row["bb_corner_index"])
    if fi not in bboxes:
        bboxes[fi] = {}
    if corner >= 0:
        bboxes[fi][corner] = (x,y,t)

# for k,v in bboxes.items():
#     print(len(v), k,v)


fig = plt.figure()
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d',computed_zorder=False)

# syntax for 3-D projection
x = events_df["x"]
y = events_df["y"]
t = events_df["t"]
c = events_df["bb_frame_index"].astype(int)
ax.scatter(x, t, y, alpha=0.8, cmap='cool', zorder=1)

# xx = np.array([0,1,0,1,0,1,0,1]) * 20 + 640
# tt = np.array([0,0,0,0,1,1,1,1]) * 20 + 100
# yy = np.array([0,0,1,1,0,0,1,1]) * 20 + 440

lines_start = [0,0,2,1,4,4,6,5,0,1,2,3]
lines_end =   [1,2,3,3,5,6,7,7,4,5,6,7]
lines_zorder= [2,2,2,2,0,0,0,0,0,0,1,1]
for k,bb in bboxes.items():
    if len(bb) != 8:
        continue
    for i in range(len(lines_start)):
        a = lines_start[i]
        b = lines_end[i]
        zo = lines_zorder[i]
        pointa = bb[a]
        pointb = bb[b]
        ax.plot([pointa[0],pointb[0]], [pointa[2],pointb[2]], [pointa[1],pointb[1]], color='black', alpha=0.5, zorder=zo)

# syntax for plotting
ax.set_xlabel('x', fontsize=22)
ax.set_ylabel('t', fontsize=22)
ax.set_zlabel('y', fontsize=22)
ax.elev = 15
ax.azim = 15
# ax.axis('off')
ax.margins(0)
plt.show()

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
fig.set_size_inches(4.7747,3.5)
plt.savefig('output/figures/3d.pgf')

