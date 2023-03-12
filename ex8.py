import numpy as np
import matplotlib.pyplot as plt
import time as time
import matplotlib

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })


def f(x):
    """
    Function to be integrated between 0 and 1
    """
    return 3 * (x ** 2)

class ImportanceSampling:
    def __init__(self,n_points):
        self.n_points = n_points
        self.x = np.linspace(0, 1, self.n_points)


    def trapezoidal(self):
        integral = np.trapz(f(self.x), self.x)
        return integral

    def mc_1(self):
        samples = np.random.random(self.n_points)
        integral = np.mean(f(samples))
        return integral

    def mc_2_weight(self,x,inverse=False):
        if inverse == False:
            return 2 * x
        else:
            return np.sqrt(x)

    def mc_3_weight(self,x,inverse=False):
        if inverse == False:
            return 4 * x**3
        else:
            return np.power(x,1/4)

    def mc_2(self):
        samples = np.random.random(self.n_points)
        biased_samples = self.mc_2_weight(samples,inverse=True)

        integral = np.mean(f(biased_samples)/self.mc_2_weight(biased_samples))
        return integral

    def mc_3(self):
        samples = np.random.random(self.n_points)
        biased_samples = self.mc_3_weight(samples, inverse=True)

        integral = np.mean(f(biased_samples) / self.mc_3_weight(biased_samples))
        return integral


if __name__ == '__main__':

    def g(x):
        return 0.05/(np.sqrt(x))

    N = 100
    points = np.linspace(10,10 + N + 1,N)
    variances = np.zeros((3,N))
    integrals = np.zeros((4,N))
    n_runs = 100
    for idx, point in enumerate(points):
        n_points = int(point)
        mc = ImportanceSampling(n_points=n_points)
        integral = np.zeros((4,n_runs))
        for i in range(n_runs):
            integral[0,i] = mc.mc_1()
            integral[1,i] = mc.mc_2()
            integral[2,i] = mc.mc_3()
            integral[3,i] = mc.trapezoidal()

        variances[0,idx] = np.var(integral[0])
        variances[1,idx] = np.var(integral[1])
        variances[2,idx] = np.var(integral[2])
        integrals[0,idx] = np.mean(integral[0])
        integrals[1,idx] = np.mean(integral[1])
        integrals[2,idx] = np.mean(integral[2])
        integrals[3,idx] = np.mean(integral[3])

    fig, ax = plt.subplots(1,2,figsize=(6.2, 3.2))
    ax[0].plot(points, integrals[0],color='b',label=r'$g(x) = 1$')
    ax[0].plot(points, integrals[1],color='#FF9500',label=r'$g(x) = 2x$')
    ax[0].plot(points, integrals[2],color='#00B945',label=r'$g(x) = 4x^3$')
    ax[0].plot(points, integrals[3],color='r',label=r'Trapezoidal')
    ax[0].set_xlabel(r'$N$')
    ax[0].set_ylabel(r'$I$')
    ax[0].set_xlim((10,110))

    ax[1].plot(points, variances[0], color='b')
    ax[1].plot(points, variances[1], color='#FF9500')
    ax[1].plot(points, variances[2], color='#00B945')
    ax[1].set_xlabel(r'$N$')
    ax[1].set_xlim((10,110))
    ax[1].set_ylabel(r'$\sigma^2_I$')
    fig.tight_layout()

    fig.legend(frameon=False,ncols=4,loc='upper center',bbox_to_anchor=(0.54,1.0))
    plt.subplots_adjust(top=0.85)

    # plt.savefig('results/importance_sampling.pgf')
    plt.show()
