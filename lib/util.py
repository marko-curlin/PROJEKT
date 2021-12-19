from pathlib import Path

import numpy as np
import open3d as o3d

from lib.constants import *


def get_ply_file_path(version):
    if version == "OG":
        return construct_path(OBJ_FOLDER, OG_PLY_FILE_NAME)
    elif version == "inliers":
        return construct_path(OBJ_FOLDER, OG_PLY_FILE_NAME)

    raise Exception("No correct version chosen")


def construct_path(*paths):
    path = Path()
    for _path in paths:
        path = path.joinpath(_path)
    return path


def ply_file_to_numpy_array(ply_file_path):
    scene = o3d.io.read_point_cloud(str(ply_file_path))
    return np.asarray(scene.points)


def point_cloud_to_ply_file(point_cloud, ply_file_path):
    if not o3d.io.write_point_cloud(str(ply_file_path), point_cloud, print_progress=True):
        raise Exception("Point cloud write failed")


def map_3d_to_plane(plane_normal, point_on_plane, point_cloud):
    mapped_points = np.empty((len(point_cloud), 3))

    for i, point in enumerate(point_cloud):
        p_ = point - point_on_plane

        p_normal = np.dot(p_, plane_normal) * plane_normal
        p_tangent = p_ - p_normal

        mapped_point = p_tangent + point_on_plane
        mapped_points[i] = mapped_point

    return mapped_points


def slice_3d_with_plane(plane_normal, point_on_plane, point_cloud, distance_to_slice=.1):
    plane_slice = np.empty((len(point_cloud), 3))

    points_in_slice_counter = 0
    for point in point_cloud:
        p_ = point - point_on_plane

        dist_to_plane = np.dot(p_, plane_normal)
        if abs(dist_to_plane) > distance_to_slice:
            continue

        p_normal = np.dot(p_, plane_normal) * plane_normal
        p_tangent = p_ - p_normal

        mapped_point = p_tangent + point_on_plane
        plane_slice[points_in_slice_counter] = mapped_point
        points_in_slice_counter += 1

    plane_slice = np.delete(plane_slice, slice(points_in_slice_counter, len(plane_slice)), axis=0)

    return plane_slice


def numpy_array_to_point_cloud_object(np_array):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np_array)
    return point_cloud
