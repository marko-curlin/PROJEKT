from lib import util

import open3d as o3d


def main():
    ply_file_path = util.get_ply_file_path(util.INLIERS)
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
