#%%
from nuscenes.nuscenes import NuScenes
import open3d as o3d
import os
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from PIL import Image

# Initialize the NuScenes dataset
currnte_directory = os.getcwd()
data_directory = os.path.join(currnte_directory, "nuscenes_data")
nusc = NuScenes(version='v1.0-mini', dataroot= data_directory, verbose=True)

# Load a sample
sample = nusc.sample[100]
#%%  Visualize the lidar point cloud data with Open3D

def get_pointcloud(sample):  #nuscenes sample
    # Get the lidar point cloud data
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_filepath = os.path.join(nusc.dataroot, lidar_data['filename'])

    # Load the point cloud
    lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
    return lidar_pointcloud

def visualize_pointcloud(lidar_pointcloud):
    # Visualize the point cloud using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_pointcloud.points.T[:, :3])
    o3d.visualization.draw_geometries([pcd])
#%% Vizualize the bounding boxes with the lidar point cloud data
def get_bounding_boxes_for_sample(sample):
    lidar_token = sample['data']['LIDAR_TOP']
    _, boxes, _ = nusc.get_sample_data(lidar_token)
    return boxes

def filter_boxes_by_classes(boxes, classes):
    filtered_boxes = [box for box in boxes if box.name in classes]
    return filtered_boxes

def visualize_sample_with_boxes(sample, classes=["vehicle.car", "vehicle.truck"], min_points=10):
    
    # Get lidar o3d point cloud
    lidar_pointcloud = get_pointcloud(sample)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_pointcloud.points.T[:, :3])
    
    # Get bounding boxes o3d objects
    boxes = get_bounding_boxes_for_sample(sample)
    filtered_boxes = filter_boxes_by_classes(boxes, ["vehicle.car", "vehicle.truck"])
    bboxes = []
    for i, box in enumerate(filtered_boxes):

        o3d_bbox = o3d.geometry.OrientedBoundingBox()
        o3d_bbox.center = box.center
        o3d_bbox.extent = box.wlh

        # q = Quaternion(box.orientation) ->they are already in the right format
        # R = q.rotation_matrix
        # o3d_bbox.R = R 
        o3d_bbox.color = (1, 0, 0)

        indices = o3d_bbox.get_point_indices_within_bounding_box(pcd.points)
        if len(indices) >= min_points:
            bboxes.append(o3d_bbox)
            print(f"Box {i} has {len(indices)} points in it and was added.")
        else:
            print(f"Box {i} has {len(indices)} points in it and was not added due to insufficient points.")


    o3d.visualization.draw_geometries([pcd, *bboxes])


# %% Visualize images.
def display_sample_images(nusc, sample):
    # Define the camera channels you want to display
    camera_channels = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    for channel in camera_channels:
        cam_token = sample['data'][channel]
        cam_data = nusc.get('sample_data', cam_token)
        cam_filepath = nusc.get_sample_data_path(cam_token)
        
        # Open and display the image
        img = Image.open(cam_filepath)
        img.show(title=channel)

# %%

# def load_point_cloud(nusc, sample):
#     lidar_token = sample['data']['LIDAR_TOP']
#     lidar_filepath = nusc.get_sample_data_path(lidar_token)
#     pc = LidarPointCloud.from_file(lidar_filepath)
#     return pc.points.T[:, :3]

# def create_bounding_box(annotation):
#     center = np.array(annotation['translation'])
#     size = np.array(annotation['size'])
#     orientation = Quaternion(annotation['rotation']).rotation_matrix
#     return center, size, orientation

# def visualize_lidar_with_boxes(nusc, sample):
#     point_cloud = load_point_cloud(nusc, sample)
#     annotations = sample['anns']
    
#     # Convert point cloud to Open3D format
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
#     # Create and add bounding boxes without transformation
#     boxes = []
#     for ann_token in annotations:
#         annotation = nusc.get('sample_annotation', ann_token)
#         center, size, orientation = create_bounding_box(annotation)
        
#         # Create oriented bounding box
#         o3d_box = o3d.geometry.OrientedBoundingBox(center, orientation, size)
#         o3d_box.color = (1, 0, 0)  # Red color for the bounding box
#         boxes.append(o3d_box)
        
#         # Debugging bounding box details
#     # Visualize
#     o3d.visualization.draw_geometries([pcd, *boxes])

# # Visualize the sample with bounding boxes
# visualize_lidar_with_boxes(nusc, sample)


