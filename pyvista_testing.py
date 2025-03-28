#%%
from __future__ import annotations
import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
from cProfile import label
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from latex2sympy2 import latex2sympy
from sympy.physics.units.quantities import Quantity
from IPython.display import Image
from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython.core.display import HTML
def css_styling():
    styles = "<style>"+ open("styles/custom.css", "r").read()+"</style>"
    return HTML(styles)
from nbconvert.preprocessors import ExecutePreprocessor
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
import numpy as np
import scipy as sc
import scipy.linalg as li
import sympy as sy
# from scipy.integrate import odeint
from sympy.parsing.latex import parse_latex
# from scipy.optimize import fsolve
from re import T
import pandas as pd
%matplotlib widget
import os
import sys
rng = np.random.default_rng(seed=0)
np.set_printoptions(suppress=True)
# import torch
# # import pytorch3d
# import mujoco as mj
# import mujoco.viewer
import numpy as np
# import mediapy as media
import matplotlib.pyplot as plt
%env MUJOCO_GL=egl
# Printing.
np.set_printoptions(precision=3, suppress=True, linewidth=100)
import vtk
from pyvista import examples
import pyvista as pv
import math
import pyvista as pv
from pyvista import examples
# pv.set_jupyter_backend('trame')
# %%
import numpy as np
points = np.random.random((1000, 3))
pc = pv.PolyData(points)
pc.plot(scalars=points[:, 2], point_size=5.0, cmap='jet')
# %%
cyl = pv.Cylinder()
arrow = pv.Arrow()
sphere = pv.Sphere()
plane = pv.Plane()
line = pv.Line()
box = pv.Box()
cone = pv.Cone()
poly = pv.Polygon()
disc = pv.Disc()
# %%
# %%
oshoe=pv.read('./RightFoot.stl')
oshoe=oshoe.scale([1000, 1000, 1000])
cpos = [
    (7.656346967151718, -9.82071079151158, -11.021236183314311),
    (22.24512272564101, -45.94554282112895, .5549738359311297),
    (-62.79216753504941, -75.13057097368635, 20.311105371647392),
]
# oshoe.plot()
# Create a voxel model of the bounding oshoe
voxels = pv.voxelize(oshoe, density=oshoe.length / 200)
voxels
# p = pv.Plotter()
# p.add_mesh(voxels, color=True, show_edges=True, opacity=0.5)
# p.add_mesh(oshoe, color="lightblue", opacity=0.5)
# p.show()
# We could even add a scalar field to that new voxel model in case we wanted to create grids for modelling. In this case, let's add a scalar field for bone density noting:
voxels["density"] = np.full(voxels.n_cells, 3.65)  # g/cc
voxels.plot(scalars="density")
# A constant scalar field is kind of boring, so let's get a little fancier by added a scalar field that varies by the distance from the bounding oshoe.
voxels.compute_implicit_distance(oshoe, inplace=True)
contours = voxels.contour(6, scalars="implicit_distance")
p = pv.Plotter()
p.add_mesh(voxels, opacity=0.25, scalars="implicit_distance")
# p.add_mesh(contours, opacity=0.5, scalars="implicit_distance")
p.show()
# %%
# oshoe=box
foot=vectors = np.vstack(
        (
            (voxels.points[:, 0]),
            (voxels.points[:, 1]),
            (voxels.points[:, 2]),
        ),
    ).T
vectors[0:130000,(1,2)]=(vectors[0:130000,(1,2)]*2)
vectors[0:130000,0]=(vectors[0:130000,0]*2)
vectors[130000:-1,2]=vectors[130000:-1,2]*-1
voxels["vectors"] = vectors
voxels.set_active_vectors("vectors")
warped = voxels.warp_by_vector()
type(warped)
p = pv.Plotter(shape=(1, 2))
p.subplot(0, 0)
p.add_text("Before warp")
p.add_mesh(voxels, color='white');
p.subplot(0, 1)
p.add_text("After warp")
p.add_mesh(warped, color='white');
p.show();
# %%
