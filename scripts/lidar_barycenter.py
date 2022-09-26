# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 12:59:47 2022

@author: dngrn
"""

import numpy as np
import matplotlib.pyplot as plt
import ot
import copy
import open3d as o3d
import HDF5Loader as loader


print('Loading hdf5 file...')
path = '../Data/'
filename = 'indoor_flying4_gt.hdf5'
full_filename = path + filename

#Call loading function to get depth maps
blended_image_rect, pose = loader.HDF5Loader(full_filename)



#Choose how many frames to average
d = 5

#Choose which frames to calculate the barycenter of
frames = range(100, 100 + 10 * d, 10)

#Initialize point cloud structure
point_clouds = []
measures_weights = []

#Loop over each frame to get point clouds
for frame in frames:
    #Take only every 30 points to downsample
    point_cloud = loader.get_points(blended_image_rect[frame], pose[frame])[0:3000:10]
    point_clouds += [point_cloud]

    #Size of each point cloud determined to turn each into a distribution
    measures_weights += [ot.unif(len(point_cloud))]



#%% Compute Free Support Barycenter
k = len(point_clouds[0])  # number of Diracs of the barycenter
X_init = point_clouds[0] # initial Dirac locations
b = np.ones((k,)) / k  # weights of the barycenter 
#(it will not be optimized, only the locations are optimized)

#Compute the barycenter (using linear programming)
X = ot.lp.free_support_barycenter(point_clouds, measures_weights, X_init, b)

#Convert to open3D point cloud format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(X)

#Save and visualize point cloud
o3d.io.write_point_cloud("Barycenter.ply", pcd)
o3d.visualization.draw_geometries([pcd])
