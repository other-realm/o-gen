#%%
import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
from cProfile import label
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from scipy.optimize import fsolve
# from latex2sympy2 import latex2sympy
from sympy.physics.units.quantities import Quantity
from IPython.display import Image
from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython.core.display import HTML
def css_styling():
    styles = "<style>"+ open("../styles/custom.css", "r").read()+"</style>"
    return HTML(styles)
from nbconvert.preprocessors import ExecutePreprocessor
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
import numpy as np
import scipy as sc
import scipy.linalg as li
import sympy as sy
from scipy.integrate import odeint
from sympy.parsing.latex import parse_latex
from re import T
import torch
import pandas as pd
%matplotlib widget
import os
import sys
import torch
# import pytorch3d
import mujoco as mj
import mujoco.viewer
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
%env MUJOCO_GL=egl
# Printing.
np.set_printoptions(precision=3, suppress=True, linewidth=100)
import pygments
from IPython.display import clear_output, HTML, display
import opensim as osim
clear_output()

def print_xml(xml_string):
  formatter = pygments.formatters.HtmlFormatter()
  lexer = pygments.lexers.XmlLexer()
  highlighted = pygments.highlight(xml_string, lexer, formatter)
  display(HTML(f"<style>{formatter.get_style_defs()}</style>{highlighted}"))

def render( model, data=None, 
            height=1024,     # (px)
            duration = 15,  # (seconds)
            framerate = 60,  # (Hz)
          ):
  if data is None:
    data = mj.MjData(model)
  frames = []
  mujoco.mj_resetData(model, data)
  with mujoco.Renderer(model) as renderer:
    while data.time < duration:
      mujoco.mj_step(model, data)
      if len(frames) < data.time * framerate:
        renderer.update_scene(data, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)
  media.show_video(frames, fps=framerate)
# %%
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
model= mj.MjModel.from_xml_path('oshoe.xml')
# Create the parent spec.
parent = mujoco.MjSpec()
body = parent.worldbody.add_body()
frame = parent.worldbody.add_frame()
site = parent.worldbody.add_site()

data = mujoco.MjData(model)
# Make renderer, render and show the pixels
render(model,data,scene_option)
# %%
mujoco.viewer.launch(model,data)
# %%
osim.GetVersionAndDate()

# %%
