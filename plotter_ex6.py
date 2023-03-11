import matplotlib.pyplot as plt
import matplotlib as matplotlib
import numpy as np
from scipy import integrate
from scipy import interpolate
from ex5 import *

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

order_params = np.array([60.05444444444444,54.665277777777774,45.848333333333336,31.5775])
temps = np.array([264,276,288,300])

phase_points = np.array([0.0,0.0,0.03,0.4,0.45,0.5])
temps_phase_diagram = np.array([264,276,288,288,276,264])
homogenous_point = np.array([0.25,300])



plt.figure(figsize=(6,4))
plt.plot(phase_points,temps_phase_diagram,color='r',marker='o',linestyle='None')
plt.plot(homogenous_point[0],homogenous_point[1],color='b',marker='o')
plt.ylabel('$T$ / K')
plt.xlabel(r'$\rho(z)$ / $\sigma^{-3}$')
plt.grid()
# plt.ylim((0,62))
# plt.xlim((250,310))
plt.tight_layout()
plt.savefig('results/phase_diag2.pgf')
plt.show()