#%%
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import open3d as o3d

class LidarDataset(Dataset):
    def __init__(self, nusc, samples, classes=["vehicle.car", "vehicle.truck"], min_points=10, voxel_size=[0.5, 0.5, 4.0], max_points_per_pillar=100, max_pillars=12000):
        """
        Initialize the dataset with necessary parameters.
        """
        self.nusc = nusc
        self.samples = samples
        self.classes = classes
        self.min_points = min_points
        self.voxel_size = voxel_size
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        lidar_pointcloud = self.get_pointcloud(sample)
        boxes = self.get_bounding_boxes_for_sample(sample)
        filtered_boxes = self.filter_boxes_by_classes(boxes)

        pcd = LidarPointCloud(lidar_pointcloud.points[:4, :]) # Keep all 4 dimensions
        bboxes = []
        labels = []

        for box in filtered_boxes:
            o3d_bbox = o3d.geometry.OrientedBoundingBox()
            o3d_bbox.center = np.array([box.center[0], box.center[1], box.center[2]])
            o3d_bbox.extent = np.array([box.wlh[0], box.wlh[1], box.wlh[2]])

            # Create an open3d point cloud to use get_point_indices_within_bounding_box
            o3d_point_cloud = o3d.geometry.PointCloud()
            o3d_point_cloud.points = o3d.utility.Vector3dVector(lidar_pointcloud.points.T[:, :3])
            indices = o3d_bbox.get_point_indices_within_bounding_box(o3d_point_cloud.points)

            if len(indices) >= self.min_points:
            # Only include boxes with at least min_points (don't train over empty boxes)
                bboxes.append(box)
                labels.append(self.classes.index(box.name))
        return pcd, bboxes, labels

    def get_pointcloud(self, sample):
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_filepath = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
        return lidar_pointcloud

    def get_bounding_boxes_for_sample(self, sample):
        lidar_token = sample['data']['LIDAR_TOP']
        _, boxes, _ = self.nusc.get_sample_data(lidar_token)
        return boxes

    def filter_boxes_by_classes(self, boxes):
        return [box for box in boxes if box.name in self.classes]

# Example of how to use this dataset
if __name__ == "__main__":
    # Set up the NuScenes dataset API
    curr_directory = os.getcwd()
    data_directory = os.path.join(curr_directory, "nuscenes_data")
    nusc = NuScenes(version='v1.0-mini', dataroot=data_directory, verbose=True)

    # Create the dataset
    dataset = LidarDataset(nusc, nusc.sample)

    # Example: Get the first sample
    pcd, bboxes, labels = dataset[0]

    # pcd.points is of shape (4, 34688) [34688 points represented as (x, y, z, intensity)]
    # bboxes is a list with 3 "nuscenes.utils.data_classes.Box" objects
    # labels is a list with 3 integers representing the class indices [0, 1, 0]

# %%
