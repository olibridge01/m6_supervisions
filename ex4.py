import numpy as np
import matplotlib.pyplot as plt
import time as time

# Program global variables; universal constants/conversions
HEADER_LINES = 9
kB = 1.3806488e-23
NA = 6.02214129e+23
kcal = 4184
angstrom = 1e-10
femto = 1e-15

def kcalmol_to_joule(num):
    return num * (kcal/NA)

class Atom:
    """
    Data for a particular atom
    """
    def __init__(self, id, type, q, x, y, z, vx, vy, vz):
        self.mass = 6.63e-26
        self.T = 179.81
        self.id = id
        self.type = type
        self.q = q

        self.pos = np.zeros(3)
        self.pos[0] = x
        self.pos[1] = y
        self.pos[2] = z

        self.vel = np.zeros(3)
        self.vel[0] = vx
        self.vel[1] = vy
        self.vel[2] = vz

        self.kinetic_energy()

    def kinetic_energy(self):
        self.ke = (1/2) * self.mass * (self.vel[0]**2 + self.vel[1]**2 + self.vel[2]**2)

class LennardJones:
    def __init__(self):
        self.epsilon = 0.24
        self.sigma = 3.405

    def lj_potential(self,r):
        if r > 3 * self.sigma:
            return 0
        else:
            return 4 * self.epsilon * ((self.sigma/r)**12 - (self.sigma/r)**6)

    def lj_force(self,r):
        if r > 3 * self.sigma:
            return 0
        else:
            return -4 * self.epsilon * (-12*((self.sigma**12)/(r**13)) + 6*((self.sigma**6)/(r**7)))

class PseudoHardSphere:
    def __init__(self):
        self.lambda_a = 49
        self.lambda_r = 50
        self.sigma = 3.305
        self.epsilon = 0.24

    def phs_potential(self,r):
        if r >= (self.lambda_r/self.lambda_a) * self.sigma:
            return 0
        elif r < (self.lambda_r/self.lambda_a) * self.sigma:
            return self.lambda_r * ((self.lambda_r/self.lambda_a)**(self.lambda_a)) * self.epsilon * ((self.sigma/r)**(self.lambda_r) - (self.sigma/r)**(self.lambda_a)) + self.epsilon

    def phs_force(self,r):
        if r >= (self.lambda_r/self.lambda_a) * self.sigma:
            return 0
        elif r < (self.lambda_r / self.lambda_a) * self.sigma:
            return -self.lambda_r * ((self.lambda_r/self.lambda_a)**(self.lambda_a)) * self.epsilon * (-self.lambda_r*((self.sigma**(self.lambda_r))/(r**(self.lambda_r + 1))) + self.lambda_a*((self.sigma**(self.lambda_a))/(r**(self.lambda_a + 1))))

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


class Exercise4:
    def __init__(self, filename, potential_type):
        self.filename = filename
        self.potential_type = potential_type

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
        self.pressure = np.zeros(self.n_samples)
        self.potential_energy = np.zeros(self.n_samples)
        self.kinetic_energy = np.zeros(self.n_samples)
        self.enthalpy = np.zeros(self.n_samples)

        self.atom_data = np.ndarray(shape=(self.n_samples, self.n_atoms), dtype=object)
        for idx, line in enumerate(file_data):
            if idx % (HEADER_LINES + self.n_atoms) > (HEADER_LINES - 1):
                mol_id, mol, mol_type, q, x, y, z, vx, vy, vz = line.split()
                self.atom_data[
                    idx // (HEADER_LINES + self.n_atoms), (idx % (HEADER_LINES + self.n_atoms)) - HEADER_LINES] = Atom(mol_id, int(mol_type), float(q), float(x), float(y), float(z), float(vx), float(vy), float(vz))

        self.volume = (self.box_length ** 3)

    def compute_pressure(self,i_sample):
        idealpart = 0
        interactingpart = 0
        for i_part, particle_1 in enumerate(self.atom_data[i_sample,:]):

            idealpart += (2 * particle_1.ke * (angstrom/femto)**2 )

            for particle_2 in self.atom_data[i_sample,i_part + 1:]:
                particle_sep = distance(particle_1, particle_2, self.box_length)
                if self.potential_type == 1:
                    lj = LennardJones()
                    interactingpart += kcalmol_to_joule(0.5 * lj.lj_force(particle_sep) * particle_sep)
                elif self.potential_type == 2:
                    phs = PseudoHardSphere()
                    interactingpart += kcalmol_to_joule(0.5 * phs.phs_force(particle_sep) * particle_sep)

        print(idealpart)
        print(interactingpart)
        self.pressure[i_sample] = idealpart + interactingpart
        self.pressure[i_sample] /= (3*self.volume*(angstrom**3))
        print(f'Pressure for sample {i_sample}: {self.pressure[i_sample]} Pa')

    def compute_potential_energy(self,i_sample):
        for i_part, particle_1 in enumerate(self.atom_data[i_sample,:]):
            for particle_2 in self.atom_data[i_sample,i_part + 1:]:
                particle_sep = distance(particle_1, particle_2, self.box_length)
                if self.potential_type == 1:
                    lj = LennardJones()
                    self.potential_energy[i_sample] += kcalmol_to_joule(lj.lj_potential(particle_sep))
                elif self.potential_type == 2:
                    phs = PseudoHardSphere()
                    self.potential_energy[i_sample] += kcalmol_to_joule(phs.phs_potential(particle_sep))
        print(f'PE: {self.potential_energy[i_sample]}')


    def compute_kinetic_energy(self,i_sample):
        for particle_1 in self.atom_data[i_sample,:]:
            self.kinetic_energy[i_sample] += (particle_1.ke * (angstrom/femto)**2)
        print(f'KE: {self.kinetic_energy[i_sample]}')


    def plot_pressures(self):
        for i_sample in range(self.n_samples):
            print(f'Sample {i_sample} / {self.n_samples}')
            self.compute_pressure(i_sample)

        print(self.pressure)
        samples = np.linspace(0,49,50)
        np.savetxt('results/ex4_pressure_phs.csv',self.pressure,delimiter=',')
        plt.plot(samples,self.pressure)
        plt.show()

    def plot_enthalpy(self):
        for i_sample in range(self.n_samples):
            print(f'Sample {i_sample} / {self.n_samples}')
            self.compute_kinetic_energy(i_sample)
            self.compute_potential_energy(i_sample)
            self.compute_pressure(i_sample)
            self.enthalpy[i_sample] = self.kinetic_energy[i_sample] + self.potential_energy[i_sample] + self.pressure[i_sample]*(self.volume*(angstrom**3))

        print(self.enthalpy)
        samples = np.linspace(0,49,50)
        np.savetxt('results/ex4_enthalpy_lj.csv',self.enthalpy,delimiter=',')
        plt.plot(samples,self.enthalpy)
        plt.show()


filename = 'md_data/pres.xyz'
ex4 = Exercise4(filename,potential_type=1)

# ex4.plot_pressures()
ex4.plot_enthalpy()








