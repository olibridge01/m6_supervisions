import numpy as np
import matplotlib.pyplot as plt
import time as time

# Number of header lines per sample in each .xyz file
HEADER_LINES = 9
kB = 1.3806488e-23

class Atom:
    """
    Data for a particular atom
    """
    def __init__(self, id, type, q, x, y, z):
        self.mass = 6.63e-26
        self.T = 179.81
        self.id = id
        self.type = type
        self.q = q

        self.pos = np.zeros(3)
        self.pos[0] = x
        self.pos[1] = y
        self.pos[2] = z

        self.assign_momentum()

    def assign_momentum(self):
        self.mom = np.zeros(3)
        beta = 1 / (kB * self.T)
        for i_dim in range(3):
            self.mom[i_dim] = np.random.normal(0, np.sqrt(self.mass / beta))

        self.kinetic_energy()

    def kinetic_energy(self):
        self.ke = (1/(2*self.mass)) * (self.mom[0]**2 + self.mom[1]**2 + self.mom[2]**2)

def lennard_jones(r):
    epsilon = 119.87
    sigma = 3.405
    if r > 3 * sigma:
        return 0
    else:
        return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

def pseudo_hardsphere(r):
    lambda_a = 49
    lambda_r = 50
    sigma = 3.405
    epsilon = 119.87
    if r >= (lambda_r/lambda_a) * sigma:
        return 0
    elif r < (lambda_r/lambda_a) * sigma:
        return lambda_r * ((lambda_r/lambda_a)**(lambda_a)) * epsilon * ((sigma/r)**(lambda_r) - (sigma/r)**(lambda_a)) + epsilon

def yukawa_debye_huckel(r,t1,t2):
    epsilon = 119.87
    sigma = 3.05
    kappa = 5
    if r > 3.5:
        return 0
    elif t1 == t2:
        return epsilon * (sigma/r) * np.exp(-kappa * (r - sigma))
    elif t1 != t2:
        return -epsilon * (sigma / r) * np.exp(-kappa * (r - sigma))

def distance(a, b, box_length):
    """
    :param a: Atom object 'a'
    :param b: Atom object 'b'
    :param box_length: side length of simulation box (to account for boundary conditions)
    :return: distance between two atoms
    """
    dx = abs((a.pos[0] - b.pos[0])*box_length)
    x = min(dx, box_length - dx)

    dy = abs((a.pos[1] - b.pos[1])*box_length)
    y = min(dy, box_length - dy)

    dz = abs((a.pos[2] - b.pos[2])*box_length)
    z = min(dz, box_length - dz)

    return np.sqrt(x ** 2 + y ** 2 + z ** 2)

def maxwell_boltzmann(beta, mass, p):
    return np.sqrt(beta/(2*np.pi*mass))*np.exp(-(beta*p**2)/(2*mass))

class EnergyCalculator:
    def __init__(self,filename,potential_type):
        self.filename = filename
        self.potential_type = potential_type

        self.parse_data()

    def parse_data(self):
        with open(self.filename) as file:
            file_data = file.readlines()

        self.n_atoms = int(file_data[3].split()[0])
        self.box_length = float(file_data[5].split()[1]) - float(file_data[5].split()[0])

        self.atom_data = np.ndarray(shape=(self.n_atoms), dtype=object)
        self.type1_atoms = 0
        self.type2_atoms = 0
        for idx, line in enumerate(file_data[HEADER_LINES:HEADER_LINES + self.n_atoms]):
            if int(line.split()[2]) == 1:
                self.type1_atoms += 1
            elif int(line.split()[2]) == 2:
                self.type2_atoms += 1
            mol_id, mol, mol_type, q, x, y, z = line.split()
            self.atom_data[idx] = Atom(int(mol_id), int(mol_type), int(q), float(x), float(y), float(z))

    def compute_potential_energy(self):
        self.potential_energy = 0

        for i_part, particle_1 in enumerate(self.atom_data):
            for particle_2 in self.atom_data[i_part+1:]:
                particle_sep = distance(particle_1, particle_2, self.box_length)
                if self.potential_type == 1:
                    self.potential_energy += lennard_jones(particle_sep)
                elif self.potential_type == 2:
                    self.potential_energy += pseudo_hardsphere(particle_sep)
                elif self.potential_type == 3:
                    if particle_sep < 3.05:
                        print('Ouch!')
                    self.potential_energy += yukawa_debye_huckel(particle_sep,particle_1.type,particle_2.type)

        print(self.potential_energy)

    def velocity_distribution(self,dim):
        moms = np.zeros((3,self.n_atoms))
        for idx, particle in enumerate(self.atom_data):
            for i_dim in range(3):
                moms[i_dim,idx] = particle.mom[i_dim]

        p = np.linspace(-1e-22,1e-22,1000)
        plt.plot(p,maxwell_boltzmann(1/(kB*179.81),6.63e-26,p),color='r')
        plt.hist(moms[dim,:],bins=100,density=True,color='b')
        # np.savetxt('results/px.csv',moms,delimiter=',')
        plt.show()

    def compute_kinetic_energy(self):
        self.kinetic_energy = 0
        for particle in self.atom_data:
            self.kinetic_energy += particle.ke/kB

    def compute_total_energy(self):
        self.compute_potential_energy()
        self.compute_kinetic_energy()
        print(self.kinetic_energy)
        print((self.kinetic_energy*(2/3))/(self.n_atoms))
        self.total_energy = self.kinetic_energy + self.potential_energy
        print(self.total_energy)


if __name__ == '__main__':
    filename = 'md_data/conf.xyz'
    pe = EnergyCalculator(filename,1)
    # pe.compute_potential_energy()
    # pe.velocity_distribution(2)
    pe.compute_total_energy()
