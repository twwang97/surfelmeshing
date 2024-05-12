import open3d as o3d
import numpy as np


# pcd_filePath = "pcd0409.ply"
# mesh_filePath = "mesh_obj_0409.obj"

# pcd_filePath = "pcdt0409.ply"
# mesh_filePath = "mesh_objt_0409.obj"

# pcd_filePath = "pcd_test03_0409.ply"
# mesh_filePath = "mesh_obj_test03_0409.obj"

pcd_filePath = "pcd_obs30_0410.ply"
mesh_filePath = "mesh_obs30_0410.obj"

print("Testing IO for point cloud ...")
pcd = o3d.io.read_point_cloud(pcd_filePath)
o3d.visualization.draw_geometries([pcd], window_name='raw pcd')

print("Testing IO for mesh ...")
mesh = o3d.io.read_triangle_mesh(mesh_filePath)
# o3d.io.write_triangle_mesh('ball_pivoting_raw.ply', mesh)
# o3d.visualization.draw_geometries([mesh], window_name='raw mesh')

mesh_vertices_np = np.asarray(mesh.vertices)
print("mesh_vertices.shape = ", mesh_vertices_np.shape)
pcd = mesh.sample_points_uniformly(number_of_points=mesh_vertices_np.shape[0])
pcd = pcd.voxel_down_sample(voxel_size=0.05)
pcd.estimate_normals()
# o3d.visualization.draw_geometries([pcd], window_name='voxel_down_sample')

print('Start to reconstruct the 3d object')
# http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
radii = [0.05, 0.1, 0.2, 0.5, 1.0]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
o3d.io.write_triangle_mesh('ball_pivoting.ply', rec_mesh)
o3d.visualization.draw_geometries([rec_mesh], window_name='ball_pivoting')
o3d.visualization.draw_geometries([pcd, rec_mesh], window_name='ball_pivoting')
