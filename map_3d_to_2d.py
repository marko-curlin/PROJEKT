import numpy as np
from lib import util


def main():
    # plane from PCA on inliers point-cloud
    normal = np.array([0.168306, 0.0300822, 0.985276])
    # p0 is a point on plane
    p0 = np.array([-27.267776, 1.233677, 11.562071])

    ply_file_path = util.get_ply_file_path("inliers")
    point_cloud = util.ply_file_to_numpy_array(ply_file_path)

    distance_to_slice = 0.15
    cloud_slice = util.slice_3d_with_plane(normal, p0, point_cloud, distance_to_slice)

    cloud_slice_object = util.numpy_array_to_point_cloud_object(cloud_slice)

    cloud_slice_output_file = util.construct_path(util.OBJ_FOLDER, f"vineyard_inliers_PCA_slice-slice_dst={distance_to_slice}.ply")
    util.point_cloud_to_ply_file(cloud_slice_object, cloud_slice_output_file)


if __name__ == '__main__':
    main()
