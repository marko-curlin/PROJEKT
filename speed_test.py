import pathlib
import time


import open3d as o3d
import numpy as np

from lib.ply import read_ply  # self written plyfile reader (source: StackOverflow)

obj_folder = 'objects'
ply_file_name = "HECTOR_2020-07-20-13-53-55-segmented_vineyard.ply"
ply_file = pathlib.Path().joinpath(obj_folder).joinpath(ply_file_name)

start = time.time()
scene2 = read_ply(ply_file)
end = time.time()

print(f"self written ply.py took {end - start} seconds to load {ply_file}")
print(f"the resulting type is {type(scene2)}")


start = time.time()
scene3 = o3d.io.read_point_cloud(str(ply_file))
end = time.time()

print(f"open3d took {end - start} seconds to load {ply_file}")
print(f"the resulting type is {type(scene2)}")

nparray = np.asarray(scene3.points)
