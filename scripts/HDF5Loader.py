# The HDF5Loader loads the data from the MVSEC dataset
import os
import h5py
import open3d as o3d
from PIL import Image
import numpy as np


def HDF5Loader(filename):

    # Check if the file exists
    if not os.path.exists(filename):
        raise ImportError('Files does not exist')

    # Load the file
    data = h5py.File(filename, 'r')
    
    # Get depth images from right LIDAR
    blended_image_rect = list(data['davis']['left']['depth_image_rect'])
    
    image = list(data['davis']['left'][''])
    
    #Get poses from vehicle odometry
    pose = list(data['davis']['left']['pose'])
    
    return blended_image_rect, pose

def get_points(depth_map, depth_map_pose):
    '''
    Function to get the point cloud from the depth images.
    Uses camera parameters to reverse the depth map projection.

    Parameters
    ----------
    depth_map : depth map
    depth_map_pose : camera pose
    Returns
    -------
    point_array : Nx3 Numpy array of point coordinates (x,y,z)
    '''
    #Remove nan's
    depth_map_norm = np.nan_to_num(depth_map, nan=0)
    
    #Scale pixel intensity to depth in meters
    depth_map_norm /= 6.24
    
    #Get dimensions of depth map
    dim_y, dim_x = np.shape(depth_map_norm)
    c_x = int(dim_x / 2)
    c_y = int(dim_y / 2)
    
    #Save normalized depth map
    #new_img = Image.fromarray(depth_map_norm)
    #new_img.save("normalized_depth_map.png")
    
    #Create Camera
    intrinsics = [346, 260, 226.38018519795807, 226.15002947047415, 173.6470807871759, 133.73271487507847]
    
    #Use open3d's point cloud from depth map functionality
    im = o3d.geometry.Image(depth_map)
    intrinsic_cam = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_cam.set_intrinsics(*intrinsics)
    default_cam = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    #Pass xyz to Open3D.o3d.geometry.PointCloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(im, default_cam)
    
    #Apply affine transformation
    pcd = pcd.transform(depth_map_pose)
    
    #Convert to Numpy array
    point_array = np.asarray(pcd.points)

    return point_array

if __name__ == '__main__':
    # Example showing how to use HD5FLoader to retrieve data streams
    print('Loading hdf5 file...')
    path = '../Data/'
    filename = 'indoor_flying4_gt.hdf5'
    full_filename = path + filename
    blended_image_rect, pose = HDF5Loader(full_filename)
    
    example_frame = 100
    example = blended_image_rect[example_frame]
    example_pose = pose[example_frame]
    
    point_array = get_points(example, example_pose)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array)
    
    
                                                     
    o3d.io.write_point_cloud("point_cloud_example.ply", pcd)
    
    #Visualize point cloud 
    o3d.visualization.draw_geometries([pcd])
    
