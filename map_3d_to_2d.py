import numpy as np
from lib import util


def main():
    # plane from PCA on inliers point-cloud
    normal = np.array((0.27795806933947287, 0.11413808831774375, 1))
    # p0 is a point on plane
    p0 = np.array((0, 0, 132.2347192775879) )

    vineyard = util.SMALL_VINEYARD

    ply_file_path = util.get_ply_file_path(vineyard)
    point_cloud = util.read_ply_file_as_numpy_array(ply_file_path)

    distance_to_slice = 0.1
    cloud_slice = util.slice_3d_with_plane(normal, p0, point_cloud, distance_to_slice)

    cloud_slice_object = util.numpy_array_to_point_cloud_object(cloud_slice)

    cloud_slice_output_file = util.construct_path(util.OBJ_FOLDER, f"vineyard_{vineyard}_linReg_slice-slice_dst={distance_to_slice}.ply")
    util.write_point_cloud_to_file(cloud_slice_object, cloud_slice_output_file)


if __name__ == '__main__':
    main()
