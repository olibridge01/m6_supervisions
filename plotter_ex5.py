import matplotlib.pyplot as plt
import matplotlib as matplotlib
import numpy as np
from scipy import integrate
from ex5 import *

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

diffA = np.loadtxt('results/ex5_msd_A.csv',delimiter=',')
diffB = np.loadtxt('results/ex5_msd_B.csv',delimiter=',')

plt.figure(figsize=(6,3))
# plt.plot(diffA[0],diffA[1],color='b')
plt.plot(diffB[0],diffB[1],color='r')
plt.ylim((0,2))
plt.xlim((0,1e-9))
plt.xlabel('$t$ / $s$')
plt.ylabel(r'$\langle r^2(t) \rangle$ / $\AA^2$')
plt.tight_layout()
plt.savefig('results/ex5_msd_B.pgf')
plt.show()