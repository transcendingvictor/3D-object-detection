#%%
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import open3d as o3d
from visualize import get_pointcloud, visualize_sample_with_boxes

# %%
from voxelization import voxelize_point_cloud  # Make sure to properly import this function

class LidarDataset(Dataset):
    def __init__(self, nusc, samples, classes=["vehicle.car", "vehicle.truck"], min_points=10, voxel_size=[0.5, 0.5, 4.0], max_points_per_pillar=100, max_pillars=12000):
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
        
        # Apply voxelization to transform point clouds to pillars
        points = np.array(lidar_pointcloud.points[:3, :].T)  # Extract x, y, z from pointcloud
        voxel_features, voxel_coords = voxelize_point_cloud(
            points, 
            self.voxel_size, 
            self.max_points_per_pillar, 
            self.max_pillars
        )
        # voxel_features = torch.Size([voxel_size, max_points_per_pillar, 3])
        # voxel_coords = torch.Size([voxel_size, 3])
        
        # Get bounding boxes and labels
        boxes = self.get_bounding_boxes_for_sample(sample)
        filtered_boxes = [box for box in boxes if box.name in self.classes]
        labels = [self.classes.index(box.name) for box in filtered_boxes if box.name in self.classes]

        # Optionally, align and filter bounding boxes based on the voxel grid here if needed
        
        return voxel_features, voxel_coords, filtered_boxes, labels

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
    
def calculate_voxel_occupation(voxel_coords, voxel_grid_size):
    unique_voxel_indices = {tuple(coord) for coord in voxel_coords}
    total_possible_voxels = np.prod(voxel_grid_size)
    return len(unique_voxel_indices) / total_possible_voxels

def calculate_iou_boxes_voxels(boxes, voxel_coords, voxel_size):
    # Assuming boxes are in the format [x, y, z, width, length, height]
    ious = []
    for box in boxes:
        # Create voxel bounding box from voxel coordinates
        voxel_min = voxel_coords * voxel_size
        voxel_max = voxel_min + voxel_size
        # Calculate IoU with each box
        box_min = np.array([box.center[0] - box.wlh[0] / 2, box.center[1] - box.wlh[1] / 2, box.center[2] - box.wlh[2] / 2])
        box_max = np.array([box.center[0] + box.wlh[0] / 2, box.center[1] + box.wlh[1] / 2, box.center[2] + box.wlh[2] / 2])
        
        # Intersection
        inter_min = np.maximum(voxel_min, box_min)
        inter_max = np.minimum(voxel_max, box_max)
        inter_size = np.maximum(inter_max - inter_min, 0)
        intersection = np.prod(inter_size)

        # Union
        box_volume = np.prod(box_max - box_min)
        voxel_volume = np.prod(voxel_max - voxel_min)
        union = box_volume + voxel_volume - intersection

        # IoU
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
    
    return np.mean(ious)

if __name__ == '__main__':
    # Initialize the NuScenes dataset
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, "nuscenes_data")
    nusc = NuScenes(version='v1.0-mini', dataroot=data_directory, verbose=True)

    # Load a sample
    sample = nusc.sample[10]

    # Create an instance of the LidarDataset
    dataset = LidarDataset(nusc, [sample])

    # Get the data for the sample
    voxel_features, voxel_coords, bboxes, labels = dataset[0]
    
    print("Voxel Features Shape:", voxel_features.shape)  # e.g., torch.Size([max_pillars, max_points_per_pillar, 3]) depending on your settings
    print("Voxel Coords Shape:", voxel_coords.shape)  # e.g., torch.Size([max_pillars, 3])
    print("Bounding Boxes len:", len(bboxes))  # List of bounding boxes for the objects in the sample
    print("Labels:", labels)  # List of labels corresponding to the classes of objects

    # Calculate and print voxel occupation
    voxel_grid_size = [torch.max(voxel_coords[:, i]).item() + 1 for i in range(3)]  # Use torch.max and convert to Python int with .item()
    voxel_occupation = calculate_voxel_occupation(voxel_coords.numpy(), voxel_grid_size)
    print("Voxel Occupation:", voxel_occupation)

    # Calculate and print average IoU
    average_iou = calculate_iou_boxes_voxels(bboxes, voxel_coords.numpy(), dataset.voxel_size)
    print("Average IoU between BBoxes and Voxels:", average_iou)

    # Visualize the LiDAR point cloud with bounding boxes
    # visualize_sample_with_boxes(sample)

    # Optionally visualize or further process the data
    # For example, visualize the first few voxels
    # import matplotlib.pyplot as plt
    # for i in range(min(5, voxel_features.shape[0])):  # Visualize the first 5 voxels
    #     plt.figure()
    #     plt.scatter(voxel_features[i, :, 0], voxel_features[i, :, 1], s=10)
    #     plt.title(f'Voxel {i} in 2D space')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.grid(True)
    #     plt.show()

# %%
# non-voxelization version
# class LidarDataset(Dataset):
#     def __init__(self, nusc, samples, classes=["vehicle.car", "vehicle.truck"], min_points=10, voxel_size=[0.5, 0.5, 4.0], max_points_per_pillar=100, max_pillars=12000):
#         """
#         Initialize the dataset with necessary parameters.
#         """
#         self.nusc = nusc
#         self.samples = samples
#         self.classes = classes
#         self.min_points = min_points
#         self.voxel_size = voxel_size
#         self.max_points_per_pillar = max_points_per_pillar
#         self.max_pillars = max_pillars

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         lidar_pointcloud = self.get_pointcloud(sample)
#         boxes = self.get_bounding_boxes_for_sample(sample)
#         filtered_boxes = self.filter_boxes_by_classes(boxes)

#         pcd = LidarPointCloud(lidar_pointcloud.points[:4, :]) # Keep all 4 dimensions
#         bboxes = []
#         labels = []

#         for box in filtered_boxes:
#             o3d_bbox = o3d.geometry.OrientedBoundingBox()
#             o3d_bbox.center = np.array([box.center[0], box.center[1], box.center[2]])
#             o3d_bbox.extent = np.array([box.wlh[0], box.wlh[1], box.wlh[2]])

#             # Create an open3d point cloud to use get_point_indices_within_bounding_box
#             o3d_point_cloud = o3d.geometry.PointCloud()
#             o3d_point_cloud.points = o3d.utility.Vector3dVector(lidar_pointcloud.points.T[:, :3])
#             indices = o3d_bbox.get_point_indices_within_bounding_box(o3d_point_cloud.points)

#             if len(indices) >= self.min_points:
#             # Only include boxes with at least min_points (don't train over empty boxes)
#                 bboxes.append(box)
#                 labels.append(self.classes.index(box.name))
#         return pcd, bboxes, labels

#     def get_pointcloud(self, sample):
#         lidar_token = sample['data']['LIDAR_TOP']
#         lidar_data = self.nusc.get('sample_data', lidar_token)
#         lidar_filepath = os.path.join(self.nusc.dataroot, lidar_data['filename'])
#         lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
#         return lidar_pointcloud

#     def get_bounding_boxes_for_sample(self, sample):
#         lidar_token = sample['data']['LIDAR_TOP']
#         _, boxes, _ = self.nusc.get_sample_data(lidar_token)
#         return boxes

#     def filter_boxes_by_classes(self, boxes):
#         return [box for box in boxes if box.name in self.classes]

# # Example of how to use this dataset
# if __name__ == "__main__":
#     # Set up the NuScenes dataset API
#     curr_directory = os.getcwd()
#     data_directory = os.path.join(curr_directory, "nuscenes_data")
#     nusc = NuScenes(version='v1.0-mini', dataroot=data_directory, verbose=True)

#     # Create the dataset
#     dataset = LidarDataset(nusc, nusc.sample)

#     # Example: Get the first sample
#     pcd, bboxes, labels = dataset[0]

#     # pcd.points is of shape (4, 34688) [34688 points represented as (x, y, z, intensity)]
#     # bboxes is a list with 3 "nuscenes.utils.data_classes.Box" objects
#     # labels is a list with 3 integers representing the class indices [0, 1, 0]