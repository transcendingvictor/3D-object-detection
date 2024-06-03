# aka pillarization.py
# This file should take pointclouds and transform them into 2D grid data with many channels
# The way voxels are defined is very simple (just storing point coordinates)  so not meaningful (statistics, embedding...)
# TODO: statistic (mean, std, max, min) not jsut the data

#%%
import numpy as np
import torch

def voxelize_point_cloud(points, voxel_size, max_points_per_voxel, max_voxels):
    """
    Converts a point cloud to a voxel grid with a fixed number of voxels.
    """
    # Define the voxel grid boundaries
    grid_bound = np.array([[np.min(points[:, i]), np.max(points[:, i])] for i in range(3)])
    voxel_grid_size = ((grid_bound[:, 1] - grid_bound[:, 0]) / np.array(voxel_size)).astype(int)

    # Calculate voxel indices for each point
    voxel_indices = np.floor((points - grid_bound[:, 0]) / np.array(voxel_size)).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, voxel_grid_size - 1)

    # Map 3D indices to a linear flat index
    voxel_index = voxel_indices[:, 0] * (voxel_grid_size[1] * voxel_grid_size[2]) + voxel_indices[:, 1] * voxel_grid_size[2] + voxel_indices[:, 2]
    unique_voxels, inverse_indices = np.unique(voxel_index, return_inverse=True, axis=0)

    # Limit the number of voxels
    unique_voxels = unique_voxels[:max_voxels]
    voxel_features = []
    voxel_coords = []

    for idx in unique_voxels:
        mask = idx == inverse_indices
        if np.any(mask):  # Check if there are any points in this voxel
            selected_points = points[mask][:max_points_per_voxel]  # Limit points per voxel
            # Pad with zeros if fewer than max_points_per_voxel
            padded_points = np.zeros((max_points_per_voxel, points.shape[1]))
            padded_points[:len(selected_points)] = selected_points
            voxel_features.append(padded_points)
            voxel_coords.append(voxel_indices[mask][0])  # All points in a voxel share the same indices
        else:
            # Handle the case where no points are in the voxel
            # Append dummy data or handle accordingly
            voxel_features.append(np.zeros((max_points_per_voxel, points.shape[1])))
            voxel_coords.append(np.zeros(3, dtype=int))  # Append a dummy voxel coordinate

    voxel_features = torch.tensor(voxel_features, dtype=torch.float32)
    voxel_coords = torch.tensor(voxel_coords, dtype=torch.int32)

    return voxel_features, voxel_coords

if __name__ == "__main__":
    # Create dummy point cloud data
    points = np.random.rand(1000, 3) * 100  # 1000 points in 3D space
    voxel_size = [5.0, 5.0, 5.0]  # Voxel size in each dimension
    max_points_per_voxel = 50
    max_voxels = 200

    voxel_features, voxel_coords = voxelize_point_cloud(points, voxel_size, max_points_per_voxel, max_voxels)
    print("Voxel Features Shape:", voxel_features.shape)
    print("Voxel Coords Shape:", voxel_coords.shape)

# %%

