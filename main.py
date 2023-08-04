import sys
import torch
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(suppress=True)
import pandas as pd
import pywavefront
import pyglet
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples
import math

mesh = pv.read("textured_output.obj")
# mesh = mesh.rotate_x(180, transform_all_input_vectors=True, inplace=True)

texture = pv.read_texture("textured_output.jpg")


def generate_points(subset=1):  # bigger subset more points ohhhh okok
    """A helper to make a 3D NumPy array of points (n_points by 3)"""
    dataset = pv.PolyData("textured_output.obj")  # this is where it is btw ohhh ok thanks
    ids = np.random.randint(low=0, high=dataset.n_points - 1, size=int(dataset.n_points * subset))
    return dataset.points[ids]


points = generate_points()

print(len(points))
# Print first 5 rows to prove its a numpy array (n_points by 3)
# Columns are (X Y Z)
point_cloud = pv.PolyData(points)  # this is where we create a point cloud
data = points[:, 1]
point_cloud["elevation"] = data #adds color
point_cloud = point_cloud.rotate_x(90, transform_all_input_vectors=True, inplace=True)

#point_cloud.plot(render_points_as_spheres=True)
print(len(points))
#mesh.plot(texture=texture)  # adds color? - no

# converts the coordinates of cloud ohhhhh okok
new = pv.convert_array(pv.convert_array(point_cloud.points))
x = new[:, 0]
x_dif = max(x) - min(x)
y = new[:, 1]
y_dif = max(y) - min(y)
z = new[:, 2]
new[:, 0] = new[:, 0] - min(x)
new[:, 1] = new[:, 1] - min(y)
new[:, 2] = new[:, 2] - min(z)
a = np.zeros((int(max(new[:, 1]) * 10)+1, int(max(new[:, 0]) * 10)+1), dtype=float)
b=np.zeros((int(max(new[:, 1]) * 10)+1, int(max(new[:, 0]) * 10)+1), dtype=float)

# print(new)

test = []
print(max(new[:, 2]))
print(a.shape)

#puts in values for z based on (x,y) coordinates
for i in range(len(new[:, 0])):
    h = new[i, 2]
    x_coord = int(new[i, 1]*10)
    y_coord = int(new[i, 0]*10)
    if (h > a[x_coord, y_coord] and h<2.35): #adding to floor map
        a[x_coord, y_coord] = h
    if (h > b[x_coord, y_coord] and  h>2.35): #adding to ceiling map
        b[x_coord, y_coord] = h

print(len(test))

# removes 0s from columns and rows
a= a[:, ~np.all(a == 0, axis=0)]
a = a[~np.all(a==0, axis=1), :]
print(b)
print(new)

np.savetxt("data8.csv", b, delimiter=',')


# graphs the point cloud and if u right click it gives u the coordinates
def callback(point):
    """Create a cube and a label at the click point."""
    pl.add_mesh(point_cloud)
    pl.add_point_labels(point, [f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"])

pl = pv.Plotter()
pl.add_mesh(point_cloud)
pl.enable_surface_point_picking(callback=callback, show_point=False)
pl.show()
# -------------------------------------------------------------------
