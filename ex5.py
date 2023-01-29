import numpy as np
import matplotlib.pyplot as plt
import time as time

def kcalmol_to_joule(num):
    return num * (kcal/NA)

# Program global variables; universal constants/conversions
HEADER_LINES = 9
kB = 1.3806488e-23
NA = 6.02214129e+23
kcal = 4184
angstrom = 1e-10
femto = 1e-15
sigma = 3.405 * angstrom
mass = 6.63e-26
epsilon = 0.24
tau = np.sqrt((mass * sigma**2)/(kcalmol_to_joule(epsilon)))


class Atom:
    """
    Data for a particular atom
    """
    def __init__(self, id, type, q, x, y, z):
        self.mass = 6.63e-26
        self.id = id
        self.type = type
        self.q = q

        self.pos = np.zeros(3)
        self.pos[0] = x
        self.pos[1] = y
        self.pos[2] = z

class Centroid:
    def __init__(self, x, y, z):
        self.pos = np.zeros(3)
        self.pos[0] = x
        self.pos[1] = y
        self.pos[2] = z

def distance(a, b, box_length):
    dx = abs(a.pos[0] - b.pos[0])
    dy = abs(a.pos[1] - b.pos[1])
    dz = abs(a.pos[2] - b.pos[2])
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

class Exercise5:
    def __init__(self, filename):
        self.filename = filename
        self.timestep_length = 2e-15
        self.n_timesteps_per_sample = 10000
        self.parse_data()

    def parse_data(self):
        """
        Loads atom data from .xyz file, and adds various simulation parameters to RDF class attributes (e.g. self.n_atoms etc.)
        """
        with open(self.filename) as file:
            file_data = file.readlines()

        self.n_atoms = int(file_data[3].split()[0])
        self.box_length = float(file_data[5].split()[1]) - float(file_data[5].split()[0])
        self.n_samples = int(len(file_data) / (HEADER_LINES + self.n_atoms))
        self.com = np.ndarray(shape=(self.n_samples), dtype=object)

        self.atom_data = np.ndarray(shape=(self.n_samples, self.n_atoms), dtype=object)
        for idx, line in enumerate(file_data):
            if idx % (HEADER_LINES + self.n_atoms) > (HEADER_LINES - 1):
                mol_id, mol, mol_type, q, x, y, z = line.split()
                self.atom_data[idx // (HEADER_LINES + self.n_atoms), (idx % (HEADER_LINES + self.n_atoms)) - HEADER_LINES] = Atom(int(mol_id), int(mol_type), float(q), float(x), float(y), float(z))

        # Sort atom data into order of mol_id to make it easier to compute MSD
        for i_sample in range(self.n_samples):
            self.atom_data[i_sample,:] = sorted(self.atom_data[i_sample,:], key=lambda Atom: Atom.id)

        # for i in range(30):
        #     print(self.atom_data[0,i].id)
        # print(self.atom_data[0,1328].pos)

    def centre_of_mass(self,i_sample):
        com = np.zeros(3)
        for i_dim in range(3):
            for i_particle in self.atom_data[i_sample,:]:
                com[i_dim] += (1/self.n_atoms) * i_particle.pos[i_dim]
        self.com[i_sample] = Centroid(com[0], com[1], com[2])

    def compute_msd(self):
        self.msd = np.zeros(self.n_samples)
        for i_sample in range(self.n_samples):
            print(f'{i_sample+1} / {self.n_samples}')
            self.centre_of_mass(i_sample)
            for idx, particle in enumerate(self.atom_data[i_sample,:]):
                vec = np.linalg.norm(particle.pos - self.atom_data[0,idx].pos - (self.com[i_sample].pos - self.com[0].pos))
                # vec = np.linalg.norm(particle.pos - self.atom_data[0,idx].pos)

                self.msd[i_sample] += (1/self.n_atoms) * (vec**2)

    def plot_msd(self):
        self.compute_msd()
        for i in range(self.n_samples):
            print(self.com[i].pos)
        print(self.msd)
        samples = np.linspace(0,49,50)
        plt.plot(samples,self.msd)
        plt.show()

    def compute_diff_coeff(self):
        self.compute_msd()
        # Compute diffusion coeff in m^2 s^-1 (SI units)
        self.D = (self.msd[self.n_samples-1] * angstrom**2) / (6 * (self.n_samples-1 * self.n_timesteps_per_sample * self.timestep_length))
        print(self.D * (tau/(sigma**2)))


filename = 'md_data/diffusionA.xyz'
ex5 = Exercise5(filename)
ex5.compute_diff_coeff()


