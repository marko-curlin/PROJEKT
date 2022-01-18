import os

import numpy as np

from lib import util


def main():
    ply_file_path = util.get_ply_file_path(util.SLICE_01_INLIERS_K_20_DEV_3)
    point_cloud = util.read_ply_file_as_numpy_array(ply_file_path)

    origin = (-28.23891965,  -3.89433624)
    direction = (-0.99870772,  0.05082218)

    angle = np.math.atan2(direction[1], direction[0])
    if angle > np.pi/2:
        angle = angle - np.pi
    angle_degrees = np.degrees(angle)

    point_cloud_2d = util.remove_z_axis(point_cloud)

    rotated_2d = util.rotate_points(point_cloud_2d, origin=origin, degrees=-angle_degrees)

    output_file_path = os.path.splitext(ply_file_path)[0] + "_rotated_2D.ply"
    util.write_numpy_array_to_file(rotated_2d, output_file_path)


if __name__ == '__main__':
    main()
