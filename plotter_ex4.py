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

pressure_lj = np.loadtxt('results/ex4_pressure_lj.csv',delimiter=',')
pressure_phs = np.loadtxt('results/ex4_pressure_phs.csv',delimiter=',')
enthalpy_lj = np.loadtxt('results/ex4_enthalpy_lj.csv',delimiter=',')
enthalpy_phs = np.loadtxt('results/ex4_enthalpy_phs.csv',delimiter=',')

def plot_pressures(pressure_lj,pressure_phs):
    samples = np.linspace(0, 490000, 50)
    plt.figure(figsize=(6,3))
    plt.plot(samples,pressure_lj,color='r',label='Lennard-Jones')
    plt.plot(samples,pressure_phs,color='b',label='Pseudo-Hard-Sphere')
    plt.ylabel(r'$P$ / Pa')
    plt.xlabel(r'Timesteps')
    plt.ylim((-0.2e+8,2e+8))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlim((0,4.9e+5))
    plt.legend(loc='upper left')
    plt.savefig('results/ex4_pressures.pgf',bbox_inches="tight")
    plt.show()

def plot_enthalpies(enthalpy_lj,enthalpy_phs):
    samples = np.linspace(0, 490000, 50)
    plt.figure(figsize=(6,3))
    plt.plot(samples,enthalpy_lj,color='r',label='Lennard-Jones')
    plt.plot(samples,enthalpy_phs,color='b',label='Pseudo-Hard-Sphere')
    plt.ylabel(r'$H$ / J')
    plt.xlabel(r'Timesteps')
    plt.ylim((-1.3e-17,2e-17))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlim((0,4.9e+5))
    plt.legend(loc='upper left')
    # plt.savefig('results/ex4_enthalpies.pgf',bbox_inches="tight")
    plt.show()


variance = 0
n_samples = 50
block_length = 10
n_blocks = n_samples / block_length
ovr_mean = np.mean(enthalpy_phs)
print(f'Mean: {ovr_mean}')
for i in range(0,n_samples,block_length):
    variance += (1/n_blocks)*(np.mean(enthalpy_phs[i:i+block_length]) - ovr_mean)**2

print(f'Variance: {variance}')
print(f'Standard Deviation: {np.sqrt(variance)}')
