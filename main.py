import torch
import numpy as np
import pandas as pd
import pywavefront
import pyglet
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples

mesh = pv.read("textured_output.obj")
#mesh = mesh.rotate_x(180, transform_all_input_vectors=True, inplace=True)

texture = pv.read_texture("textured_output.jpg")
# print(type(texture))
# surf = mesh.extract_surface().triangulate()
# surf = surf.decimate_pro(0.9)
# print(surf.points)
# # warped = mesh.warp_by_scalar('Elevation')
# # surf = warped.extract_surface().triangulate()
# # surf = surf.decimate_pro(0.75)  # reduce the density of the mesh by 75%

# Define some helpers - ignore these and use your own data.
def generate_points(subset=.1):#bigger subset more points ohhhh okok
    """A helper to make a 3D NumPy array of points (n_points by 3)"""
    dataset = pv.PolyData("textured_output.obj")#this is where it is btw ohhh ok thanks
    ids = np.random.randint(low=0, high=dataset.n_points - 1, size=int(dataset.n_points * subset))
    return dataset.points[ids]




points = generate_points()
print(len(points))
# Print first 5 rows to prove its a numpy array (n_points by 3)
# Columns are (X Y Z)
points[0:5, :]




point_cloud = pv.PolyData(points) #this is where we create a point cloud

data = points[:, 1]
point_cloud["elevation"] = data
point_cloud = point_cloud.rotate_x(90, transform_all_input_vectors=True, inplace=True)

point_cloud.plot()
# point_cloud.plot(render_points_as_spheres=True)
print(len(points))
#mesh.plot(texture=texture)