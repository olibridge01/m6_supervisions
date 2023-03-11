import numpy as np
import matplotlib.pyplot as plt
import time as time

# Take program start time
start = time.time()

# Number of header lines per sample in each .xyz file
HEADER_LINES = 9

class Atom:
    """
    Data for a particular atom
    """
    def __init__(self, id, type, q, x, y, z):
        self.id = id
        self.type = type
        self.q = q
        self.x = x
        self.y = y
        self.z = z

def distance(a, b, box_length):
    """
    :param a: Atom object 'a'
    :param b: Atom object 'b'
    :param box_length: side length of simulation box (to account for boundary conditions)
    :return: distance between two atoms
    """
    dx = abs((a.x - b.x)*box_length)
    x = min(dx, box_length - dx)

    dy = abs((a.y - b.y)*box_length)
    y = min(dy, box_length - dy)

    dz = abs((a.z - b.z)*box_length)
    z = min(dz, box_length - dz)

    return np.sqrt(x ** 2 + y ** 2 + z ** 2)

def spherical_vol(r,dr):
    """
    :param r: radius
    :param dr: shell thickness
    :return: shell volume
    """
    r_1 = r
    r_2 = r_1 + dr
    return (4/3)*np.pi*(r_2**3 - r_1**3)
    # return 4 * np.pi * (r ** 2) * dr

class RDF:
    """
    RDF class for computing and plotting the radial distribution function from atomic position data
    """
    def __init__(self,filename,type1,type2,n_points=100):
        self.filename = filename
        self.type1 = type1
        self.type2 = type2
        self.n_points = n_points
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

        self.type1_atoms = 0
        for idx, line in enumerate(file_data[HEADER_LINES:HEADER_LINES+self.n_atoms]):
            if int(line.split()[2]) == self.type1:
                self.type1_atoms += 1

        self.atom_data = np.ndarray(shape=(self.n_samples, self.n_atoms), dtype=object)
        for idx, line in enumerate(file_data):
            if idx % (HEADER_LINES + self.n_atoms) > (HEADER_LINES - 1):
                mol_id, mol, mol_type, q, x, y, z = line.split()
                self.atom_data[idx // (HEADER_LINES + self.n_atoms), (idx % (HEADER_LINES + self.n_atoms)) - HEADER_LINES] = Atom(mol_id, int(mol_type), float(q), float(x), float(y), float(z))

        if self.type1 == 0:
            self.type1_atoms = self.n_atoms

        self.avg_density = (self.type1_atoms) / (self.box_length**3)

    def calculate_rdf(self):
        """
        Compute radial distribution function from atomic data
        """
        max_r = self.box_length / 2
        dr = max_r / self.n_points

        self.rdf = np.zeros(self.n_points)
        shell_volumes = np.zeros(self.n_points)
        self.r_axis = np.zeros(self.n_points)

        for i_point in range(self.n_points):
            shell_volumes[i_point] = spherical_vol(i_point * dr, dr)
            self.r_axis[i_point] = i_point * dr

        for i_sample in range(self.n_samples):
            print(f'{i_sample} / {self.n_samples}')

            for i_part, particle_1 in enumerate(self.atom_data[i_sample]):
                if particle_1.type == self.type1 or self.type1 == 0:
                    for particle_2 in self.atom_data[i_sample,i_part+1:]:
                        if particle_2.type == self.type2 or self.type2 == 0:
                            particle_sep = distance(particle_1,particle_2,self.box_length)
                            point_number = int(particle_sep / dr)
                            if 0 <= point_number < self.n_points:
                                if self.type1 == self.type2:
                                    self.rdf[point_number] += 2
                                else:
                                    self.rdf[point_number] += 1

        for idx, value in enumerate(self.rdf):
            self.rdf[idx] /= (shell_volumes[idx] * self.avg_density * self.type1_atoms * self.n_samples)

    def plot_rdf(self):
        """
        Plot radial distribution function
        """
        print(self.rdf)
        print(self.type1_atoms)
        plt.plot(self.r_axis,self.rdf)
        # np.savetxt('results/water_oh_1.csv',self.rdf,delimiter=',')
        # np.savetxt('results/water_oh_1_r.csv',self.r_axis,delimiter=',')
        plt.show()


# Call RDF class and plot radial distribution function
# filename = "md_data/ideal.xyz"
# ideal_gas = RDF(filename,0,0,n_points=400)
# ideal_gas.calculate_rdf()
# ideal_gas.plot_rdf()
#
# # Measure program runtime
# end = time.time()
# print(f'Runtime: {end-start}')