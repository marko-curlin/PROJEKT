from lib import util

import open3d as o3d


def find_plane(point_cloud, distance_threshold=0.2, num_iterations=500):
    plane_model, _ = point_cloud.segment_plane(distance_threshold=distance_threshold,
                                               ransac_n=3,
                                               num_iterations=num_iterations)

    plane_normal = util.get_normal(plane_model[:3])
    point_on_plane = util.get_point_on_cloud(plane_model[:3], plane_model[3])

    return plane_normal, point_on_plane


def main():
    ply_file_path = util.get_ply_file_path(util.SMALL_VINEYARD)
    point_cloud = util.read_ply_file_as_o3d_point_cloud(ply_file_path)

    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=500)  # i bez seeda u prosjeku daje isti rezultat

    print(f'plane coefficient: {plane_model[:3]}')
    print(f'plane offset: {plane_model[3]}')

    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)

    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


if __name__ == '__main__':
    main()
