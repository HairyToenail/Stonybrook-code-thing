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
def generate_points(subset=.005):#bigger subset more points ohhhh okok
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
# point_cloud.plot()
# point_cloud.plot(render_points_as_spheres=True)
print(len(points))
# mesh.plot(texture=texture)
print(point_cloud)
heights = point_cloud.points[:, 2]

std_dev = np.std(heights)


print(heights)
x = point_cloud.points[:, 0]
x.sort()
x_dif = x[-1]-x[0]
y=point_cloud.points[:, 1]
y.sort()
y_dif = y[-1]-y[0]
# a = np.array([x_dif*100000, y_dif*100000])

# Make empty array - idk if this works
# a = np.zeros(shape=(len(mesh.points[:, 0], len(mesh.points[:, 1]))


z = point_cloud.points[:, 2]

print(a)

# for i in range(len(mesh.points[:, 0])):
#     for p in range(len(mesh.points[:, 1])):
for i in range(x_dif*100000):
    for p in range(y_dif*100000):
        if(point_cloud[0].index(i/100000)>=0 and point_cloud[1].index(p/100000)>=0):
            # a[i][p]=z[point_cloud.index(#need to generate points)]
