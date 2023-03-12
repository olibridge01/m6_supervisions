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


class MC_PhotonGas:
    def __init__(self, n_vals, n_mc_steps):
        self.n_vals = n_vals
        self.n_mc_steps = n_mc_steps
        self.vals = np.linspace(0.1,2,self.n_vals)

    def metropolis_mc(self, n_steps, beta_epsilon, erroneous=False):
        currentnj = 1
        sumnj = 0
        for i in range(n_steps):
            rand_1 = np.random.random()
            if rand_1 < 0.5:
                if currentnj != 0:
                    currentnj -= 1
                acc = True
            else:
                rand_2 = np.random.random()
                if rand_2 < np.exp(-beta_epsilon):
                    currentnj += 1
                    acc = True
                else:
                    acc = False

            if acc == True or erroneous == False:
                sumnj += currentnj

        avgnj = sumnj / n_steps
        theo = 1 / (np.exp(beta_epsilon) - 1)
        error = np.absolute((theo - avgnj) / theo) * 100
        return avgnj, theo, error

    def compute_results(self,erroneous=False):
        self.nj = np.zeros(self.n_vals)
        self.theos = np.zeros(self.n_vals)
        self.errors = np.zeros(self.n_vals)

        for i, val in enumerate(self.vals):
            self.nj[i], self.theos[i], self.errors[i] = self.metropolis_mc(n_steps=self.n_mc_steps, beta_epsilon=val, erroneous=erroneous)

    def plot_results(self,j):
        fig, (ax, ax2) = plt.subplots(1,2)
        fig.set_figheight(2.7)
        fig.set_figwidth(6.2)
        ax.plot(self.vals, self.theos, color='r', label='Theoretical value')
        ax.plot(self.vals, self.nj, color='b', label='Metropolis algorithm')
        ax.legend(frameon=False)
        ax.set_xlim((0,2))
        ax.set_ylim((0,11))
        ax.set_xlabel(r'$\beta\epsilon$')
        ax.set_ylabel(r'$\langle n \rangle$')

        ax2.plot(self.vals,self.errors,color='r')
        ax2.set_xlabel(r'$\beta\epsilon$')
        ax2.set_ylabel(r'abs % error')
        ax2.set_xlim((0,2))
        ax2.set_ylim((0,100))
        ax.text(1.0,4.5,f'$N$ = {j}')
        fig.tight_layout()
        # plt.savefig('results/mc_100000steps_erroneous.pgf')
        plt.show()

if __name__ == '__main__':

    mc_code = MC_PhotonGas(100,10000)
    mc_code.compute_results(erroneous=True)
    mc_code.plot_results(100000)



