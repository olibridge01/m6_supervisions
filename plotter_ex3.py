import matplotlib.pyplot as plt
import matplotlib as matplotlib
import numpy as np
from scipy import integrate
from ex3 import *

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

moms = np.loadtxt('results/px.csv',delimiter=',')

print(moms)
plt.figure(figsize=(3,1.5))
p = np.linspace(-1e-22, 1e-22, 1000)
plt.plot(p,maxwell_boltzmann(1/(kB*179.81),6.63e-26,p),color='r')
plt.hist(moms[0,:],bins=100,density=True,color='b')
plt.xlim((-0.5e-22,0.5e-22))
plt.xlabel(r'$p$ / $kg\:m\:s^{-1}$')
plt.ylabel(r'$f(p)$')
plt.savefig('results/px2.pgf',bbox_inches="tight")
plt.show()
