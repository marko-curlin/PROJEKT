from pathlib import Path

import numpy as np
import open3d as o3d

from constants import *


def get_ply_file_path(version):
    if version == "OG":
        return Path(OBJ_FOLDER).joinpath(OG_PLY_FILE_NAME)
    elif version == "inline":
        return Path(OBJ_FOLDER).joinpath(OG_PLY_FILE_NAME)


def ply_file_to_numpy_array(ply_file_path):
    scene = o3d.io.read_point_cloud(str(ply_file_path))
    return np.asarray(scene.points)
