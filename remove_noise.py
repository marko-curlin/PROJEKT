from lib import util

import open3d as o3d

from os import path


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],)


def main():
    ply_file_path = util.get_ply_file_path(util.SLICE_01)
    point_cloud = util.read_ply_file_as_o3d_point_cloud(ply_file_path)

    nr_neighbours, std_ratio = 20, 3
    inlier_cloud, indices = point_cloud.remove_statistical_outlier(nb_neighbors=nr_neighbours, std_ratio=std_ratio)

    # display_inlier_outlier(point_cloud, indices)

    output_ply_file = util.construct_path(path.splitext(ply_file_path)[0] + f'_inliers-k={nr_neighbours}-std_dev={std_ratio}.ply')
    util.write_point_cloud_to_file(inlier_cloud, output_ply_file)


if __name__ == '__main__':
    main()
