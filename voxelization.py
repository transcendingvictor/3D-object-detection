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

    Args:
        points (numpy.ndarray): A 2D array of shape (N, 3) representing the point cloud,
            where N is the number of points and each point has 3 coordinates (x, y, z).
        voxel_size (list or tuple): A list or tuple of length 3 specifying the size of each
            voxel in the x, y, and z dimensions.
        max_points_per_voxel (int): The maximum number of points that can be assigned to a
            single voxel. If a voxel contains more points than this limit, the excess points
            are discarded.
        max_voxels (int): The maximum number of voxels to generate. If the number of unique
            voxels exceeds this limit, only the first `max_voxels` voxels are retained.

    Returns:
        tuple: A tuple containing two tensors:
            - voxel_features (torch.Tensor): A 3D tensor of shape (num_voxels, max_points_per_voxel, 3)
                representing the features of each voxel. Each voxel contains up to `max_points_per_voxel`
                points, and each point has 3 coordinates (x, y, z). If a voxel has fewer points than
                `max_points_per_voxel`, the remaining entries are filled with zeros.
            - voxel_coords (torch.Tensor): A 2D tensor of shape (num_voxels, 3) representing the integer
                coordinates of each voxel in the voxel grid.
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
    
    # Initialize the arrays to hold the voxel data
    voxel_features = np.zeros((len(unique_voxels), max_points_per_voxel, points.shape[1]), dtype=np.float32)
    voxel_coords = np.zeros((len(unique_voxels), 3), dtype=int)

    for i, idx in enumerate(unique_voxels):
        mask = idx == inverse_indices
        if np.any(mask):
            selected_points = points[mask][:max_points_per_voxel]  # Limit points per voxel
            num_points = len(selected_points)
            voxel_features[i, :num_points] = selected_points
            voxel_coords[i] = voxel_indices[mask][0]  # All points in a voxel share the same indices

    # Convert arrays to tensors
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
    print("Voxel Features Shape:", voxel_features.shape) # torch.Size([200, 50, 3])
    print("Voxel Coords Shape:", voxel_coords.shape) #torch.Size([200, 3])

# %%

