from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import open3d as o3d

# # Generate some n x 3 matrix using a variant of sync function.
# x = np.linspace(-3, 3, 201)
# mesh_x, mesh_y = np.meshgrid(x, x)
# z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
# z_norm = (z - z.min()) / (z.max() - z.min())
# xyz = np.zeros((np.size(mesh_x), 3))
# xyz[:, 0] = np.reshape(mesh_x, -1)
# xyz[:, 1] = np.reshape(mesh_y, -1)
# xyz[:, 2] = np.reshape(z_norm, -1)
# print("Printing numpy array used to make Open3D pointcloud ...")
# print(xyz)

# trajectory_filepath = "output/open3d test/0_bee_pts22690_start16708.csv"
# trajectory_filepath = "output/open3d test/12_bee_pts4069_start6239878.csv"
# trajectory_filepath = "output/open3d test/2_but_pts186800_start9271072/frag_37.csv"
trajectory_filepath = "output/open3d test/6_dra_pts78465_start29351165/frag_2.csv"
# trajectory_filepath = "output/open3d test/21_dra_pts55549_start103333374/frag_3.csv"

# Load point cloud
df = pd.read_csv(trajectory_filepath, sep=',', header="infer")
df = df.iloc[:,:3]
print(df)

# Scale t again! Applied on first t scale!
SECOND_T_SCALE = 1.0
# df["t"] = df["t"].apply(lambda x: x*0.5)

xyz = df.to_numpy(dtype=np.float32)
print(xyz.shape[0], "points")

xyz = xyz * np.array([1.0, 1.0, SECOND_T_SCALE])

# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize.
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# Add color and estimate normals for better visualization.
pcd.paint_uniform_color([0.0, 0.2, 1.0])

# SOR
# sor_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
# sor_pcd.translate((100, 0, 0))
# print("SOR 10 2.0:", len(sor_pcd.points), "points")

sor_params = [
    (5,4.0),
    (5,6.0),
    (10,4.0),
    (10,6.0),
    (20,4.0),
    (20,6.0),
    (40,4.0),
    (40,6.0),
]
radius_params = [
    (5,4.0),
    (5,8.0),
    (5,16.0),
    (10,4.0),
    (10,8.0),
    (10,16.0),
    (20,4.0),
    (20,8.0),
    (20,16.0),
]

sor_pcds = []
for i,para in enumerate(radius_params):
    # sor_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=para[0], std_ratio=para[1])
    sor_pcd, ind = pcd.remove_radius_outlier(nb_points=para[0], radius=para[1])
    sor_pcd.translate(((i+1)*100, 0, 0))
    sor_pcds.append(sor_pcd)
    print(f"SOR {para[0]} {para[1]}: {len(sor_pcd.points): >5} points")


# Farthes point sampling; After SOR
# sampled_pcd = sor_pcd.farthest_point_down_sample(1000)
# # sampled_pcd = pcd.select_by_mask(sampled_pcd)
# sampled_pcd.translate((100, 0, 0))


# Make BBox and best fit BBox around this pointcloud
# axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
# axis_aligned_bounding_box.color = (1, 0, 0)
# oriented_bounding_box = pcd.get_oriented_bounding_box()
# oriented_bounding_box.color = (0, 1, 0)

# DBSCAN
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(pcd.cluster_dbscan(eps=10, min_points=5, print_progress=True))
# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

pcds = [pcd] + sor_pcds

for i,pcd in enumerate(pcds):
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(pcd.points) * np.array([1.0, 1.0, 1/SECOND_T_SCALE]) )


# Visualize
o3d.visualization.draw_geometries(pcds)
# o3d.visualization.draw(
#     [pcd, axis_aligned_bounding_box, oriented_bounding_box])

# Convert Open3D.o3d.geometry.PointCloud to numpy array.
# xyz_converted = np.asarray(pcd.points)
# print("Printing numpy array made using Open3D pointcloud ...")
# print(xyz_converted)

print("End.")