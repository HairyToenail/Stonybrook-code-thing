import sys
import torch
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
import pandas as pd
import pywavefront
import pyglet
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples
import math

mesh = pv.read("textured_output.obj")
#mesh = mesh.rotate_x(180, transform_all_input_vectors=True, inplace=True)

texture = pv.read_texture("textured_output.jpg")
def generate_points(subset=.05):#bigger subset more points ohhhh okok
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



# point_cloud.plot(render_points_as_spheres=True)
print(len(points))
# mesh.plot(texture=texture)

#converts the coordinates of cloud ohhhhh okok
point_cloud = pv.convert_array(pv.convert_array(point_cloud.points))
x = point_cloud[:, 0]
x_dif = max(x)-min(x)
y=point_cloud[:, 1]
y_dif = max(y)-min(y)
a = np.empty((int(x_dif*10), int(y_dif*10)), dtype=float)
z = point_cloud[:, 2]
point_cloud[:, 0] = point_cloud[:, 0]-min(x)
point_cloud[:, 1] = point_cloud[:, 1]-min(y)
point_cloud[:, 2] = point_cloud[:, 2]-min(z)
# print(point_cloud)
for i in range(int(x_dif*10)):
    if(z[int(i/10)]>=a[int(x[int(i/10)]), int(y[int(i/10)])]):
        a[int(x[int(i/10)]), int(y[int(i/10)])] = z[int(i/10)]
# print(a)

DF = pd.DataFrame(a)

# save the dataframe as a csv file
DF.to_csv("data1.csv")



#graphs the point cloud and if u right click it gives u the coordinates
def callback(point):
    """Create a cube and a label at the click point."""
    pl.add_mesh(point_cloud)
    pl.add_point_labels(point, [f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"])



pl = pv.Plotter()
pl.add_mesh(point_cloud)
pl.enable_surface_point_picking(callback=callback, show_point=False)
pl.show()
#-------------------------------------------------------------------