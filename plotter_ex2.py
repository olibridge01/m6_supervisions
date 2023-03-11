import matplotlib.pyplot as plt
import matplotlib as matplotlib
import numpy as np
from scipy import integrate
from ex2 import *

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
idealgas = np.loadtxt('results/ideal_gas_rdf.csv',delimiter=',')
idealgas_r = np.loadtxt('results/ideal_gas_r.csv',delimiter=',')

data1 = np.loadtxt('results/set1.csv',delimiter=',')
x1 = np.loadtxt('results/set1_r.csv',delimiter=',')
data2 = np.loadtxt('results/set2.csv',delimiter=',')
x2 = np.loadtxt('results/set2_r.csv',delimiter=',')
data3 = np.loadtxt('results/set3.csv',delimiter=',')
x3 = np.loadtxt('results/set3_r.csv',delimiter=',')

horiz = np.zeros(400)
for i in range(400):
    horiz[i] = 1

plt.figure(figsize=(7,3))
# plt.plot(x1,data1,color='b')
# plt.plot(x2,data2,color='b')
# plt.plot(x3,data3,color='b')
plt.plot(x3,horiz,color='r',linestyle='dashed')
plt.plot(idealgas_r,idealgas,color='b')
plt.xlabel(r'$r$ / $\AA$')
plt.ylabel(r'$g(r)$')
plt.ylim((0.95,1.1))
plt.xlim((0,30))
# plt.savefig('results/idealgas.png',bbox_inches="tight")
plt.show()

# ideal_gas = RDF('md_data/set3.xyz',n_points=200)
# def f(density,datapoint,r):
#     return 4*np.pi*density*datapoint*(r**2)
#
# intdata = np.zeros(200)
# for i in range(200):
#     intdata[i] = f(ideal_gas.avg_density,data2[i],x2[i])
#
# y_int = integrate.cumtrapz(intdata,x2,initial=0)
#
# print(y_int[48],x2[48])