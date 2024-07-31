"""
example of xml:
    ...
    <shape type="sphere">
        <float name="radius" value="0.015"/>
        <transform name="toWorld">
            <translate x="0.08808434009552002" y="0.3710571527481079" z="0.09070360660552979"/>
            <scale value="0.7"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="1.0,0.0,1.0"/>
        </bsdf>
    </shape>
    ...
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import bee_utils as bee

# INPUT_DIR = Path("../../code/foldingnet2/mitsuba/Reconstruct_foldnet_plane/shapenetcorev2/test0/")
INPUT_DIR = Path("../../code/foldingnet2/mitsuba/Reconstruct_insect_foldnet_gaussian_k20_e1600/insect/test487/")
OUTOUT_BASE_DIR = Path("output/mitsuba_xml_to_csv/")

if __name__ == '__main__':
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    input_dir = INPUT_DIR
    output_dir = OUTOUT_BASE_DIR / f"{input_dir.parent.name}_{input_dir.name}_{datetime_str}"

    for fpath in input_dir.glob("*.xml"):

        # [ [x:float,y:flaot,z:float], ... ]
        points = []

        with open(fpath) as f:
            for line in f:
                if "<translate x=" in line:
                    # is coords ine
                    arr = line.split(" ")
                    coord_strs = arr[-3:]
                    xyz = [0.0]*3
                    for i,coord_str in enumerate(coord_strs):
                        coord = float(coord_str.split("\"")[1])
                        xyz[i] = coord
                    points.append(xyz)

        # Save
        filestem = fpath.name.replace(".xml","")
        output_path = output_dir / f"{filestem}.csv"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame(points, columns =["x","y","z"]) 
        df.to_csv(output_path, index=False, header=True, decimal='.', sep=',', float_format='%.6f')
        print("Saved", output_path)