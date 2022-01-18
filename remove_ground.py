import numpy as np

from lib import util


def main():
    # plane from PCA on inliers point-cloud
    normal = np.array([0.27795807, 0.11413809, 1])
    # p0 is a point on plane
    p0 = np.array([0, 0, 132.2347192775879])

    ply_file_path = util.get_ply_file_path(util.INLIERS)
    data_points = util.read_ply_file_as_numpy_array(ply_file_path)

    max_distance = 0.6
    points_above_plane = util.get_points_above_plane(normal, p0, data_points, max_distance=max_distance)

    point_cloud_object = util.numpy_array_to_point_cloud_object(points_above_plane)

    # util.draw_cloud(point_cloud_object)
    cloud_slice_output_file = util.construct_path(util.OBJ_FOLDER, f"vineyard_inliers_above_ground-max_dst={max_distance}.ply")
    util.write_point_cloud_to_file(point_cloud_object, cloud_slice_output_file)


if __name__ == '__main__':
    main()
