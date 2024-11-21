# POINT_CLOUD_PROCESSING

Enhält Skripte zu Aufbereitung der DVS-Insektenaufnahmen.

## Umgebung
Alle Skripte laufen mit Python. Für Bibliotheken kann Conda (miniconda) verwendet werden. Die installierten Packages und Versionen aus der verwendeten Umgebung sind in der Datei conda-packages.txt enthalten. Darunter python=3.9, pandas=1.4.2, scikit-learn=1.1.1, numpy=1.22.4.


## Optionen/Parameter
Die Skripte können über Konstanten (z.B. ```EVENTS_PER_FRAGMENT = 4096```) konfiguriert werden. Bei manchen Skripten gibt es je nach Eingabedaten unterschiedliche Einstellungsblöcke, die man auskommentieren kann, da die Eingabedaten unterschiedliche Formate aufweisen.

```
############### PF #############
# Für Daten/Aufnahmen von HSNR bzw. Pohle-Fröhlich.
...
##################################

############### MB #############
# Für Daten/Aufnahmen von Marc Böge.
...
##################################

############### MU #############
# Für Daten/Aufnahmen der Uni Münster.
FPS = 100
EVENTS_CSV_T_COL = 2
EVENTS_CSV_P_COL = 3
# Paths
EVENTS_CSV_DIR = Path("output/mu_h5_to_csv")
EVENTS_CSV_FILENAME = "{filestem}.csv"
...
##################################
```

## Verzeichnisse
In **output/** werden die erzeugten Daten abgelegt. Da das Verzeichnis über 50GB groß ist, ist es nicht mit im Git Repository enthalten.

- output/extracted_trajectories: Enthält extrahierte Flugbahnen aus dem extract_individual_trajectories.py Skript.
- output/csv_to_video: Enthält zu Video konvertierte DVS Aufnahmen.
- output/video_annotations: Enthält Bounding Box Annotationen zu den Videos. Format: (Frame_index, Klasse, instance_id, is_difficult, x, y, w, h).


## Skripte

### bee_utils.py
Defines common util constanct, mappings, functions and classes for several scripts.

### extract_individual_trajectories.py
Extract individual insect flight paths from a point cloud from a DVS sensor in CSV format.
For each path create a separate csv file containing the (x, y, t)-Points.

Input:
- CSV with events. Format for events: (x, y, t_in_us_or_ms, p)
- Labels (bounding boxes) from DarkLabel software. Format: (frame_index, classname, instance_id, is_difficult, x, y, w, h)

### fragment_trajectories_by_time.py
Fragment individual flight trajectories into smaller parts, so that they will fit into PointNet, which requires 4096 points.
Each part has the same time length. 
Points will be reduced or created to achieve exactly 4096 points.
Applies noise reduction (SOR or none), sampling (random or farthest-point or none) and normalization.

### tsne_inspector.py
Visualisert Features aus dem PointNet++ als t-SNE Plot.


