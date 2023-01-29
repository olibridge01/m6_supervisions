import matplotlib.pyplot as plt
import matplotlib as matplotlib
import numpy as np
from scipy import integrate
from ex3 import *

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

diffA = np.loadtxt('results/.csv',delimiter=',')
diffB = np.loadtxt('results/.csv',delimiter=',')

plt.plot()