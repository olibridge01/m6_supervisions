import numpy as np
import matplotlib.pyplot as plt
import time as time
import matplotlib

# Program global variables; universal constants/conversions
HEADER_LINES = 9
kB = 1.3806488e-23
NA = 6.02214129e+23
kcal = 4184
angstrom = 1e-10
femto = 1e-15
sigma = 3.4

def kcalmol_to_joule(num):
    return num * (kcal/NA)

class Atom:
    """
    Data for a particular atom
    """
    def __init__(self, atom_id, mol_id, type, x, y, z):
        self.mass = 6.63e-26
        self.atom_id = atom_id
        self.mol_id = mol_id
        self.type = type

        self.pos = np.zeros(3)
        self.pos[0] = x
        self.pos[1] = y
        self.pos[2] = z

def distance(a, b, box_length, box_height):
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

    dz = abs((a.pos[2] - b.pos[2])*box_height)
    z = min(dz, box_height - dz)

    return np.sqrt(x ** 2 + y ** 2 + z ** 2)

def is_contact(a,b,box_length,box_height):
    r_c = 4.1
    # Do I need to avoid double counting the inter-polymer contacts?
    # print(f'Checking {a.atom_id} with {b.atom_id}: ',end='')
    if b.atom_id - a.atom_id == 1 and a.mol_id == b.mol_id:
        # print(f'Not allowed. Bonded atoms')
        return 0
    if distance(a,b,box_length,box_height) < r_c:
        if a.mol_id != b.mol_id:
            # print(f'Match {a.atom_id} with {b.atom_id}: +2')
            return 2
        else:
            # print(f'Match {a.atom_id} with {b.atom_id}: +1')
            return 1
    else:
        # print(f'Failure')
        return 0

class Exercise6:
    def __init__(self, filename):
        self.filename = filename
        self.polymer_length = 40
        self.parse_data()

    def parse_data(self):
        """
        Loads atom data from .xyz file, and adds various simulation parameters to RDF class attributes (e.g. self.n_atoms etc.)
        """
        with open(self.filename) as file:
            file_data = file.readlines()

        self.n_atoms = int(file_data[3].split()[0])
        self.n_polymers = int(self.n_atoms / self.polymer_length)
        self.box_length = float(file_data[5].split()[1]) - float(file_data[5].split()[0])
        self.box_height = float(file_data[7].split()[1]) - float(file_data[7].split()[0])
        self.n_samples = int(len(file_data) / (HEADER_LINES + self.n_atoms))
        # print(len(file_data))

        self.atom_data = np.ndarray(shape=(self.n_samples, self.n_atoms), dtype=object)
        for idx, line in enumerate(file_data):
            if idx % (HEADER_LINES + self.n_atoms) > (HEADER_LINES - 1):
                atom_id, mol_id, mol_type, x, y, z = line.split()
                self.atom_data[idx // (HEADER_LINES + self.n_atoms), (idx % (HEADER_LINES + self.n_atoms)) - HEADER_LINES] = Atom(int(atom_id), int(mol_id), int(mol_type), float(x), float(y), float(z))

        self.volume = (self.box_length**2 * self.box_height)

    def compute_order_parameter(self):
        self.N_c = 0
        for i_sample in range(self.n_samples):
            print(f'{i_sample} / {self.n_samples}')
            for i_polymer in range(self.n_polymers):
                for idx, particle_1 in enumerate(self.atom_data[i_sample,i_polymer*self.polymer_length:(i_polymer+1)*self.polymer_length]):
                    for particle_2 in self.atom_data[i_sample,i_polymer*self.polymer_length + idx + 1:]:
                        self.N_c += is_contact(particle_1, particle_2, self.box_length, self.box_height)
        self.N_c /= (self.n_samples * self.n_polymers)
        print(self.N_c)

    def compute_phase_diagram(self,n_bins):
        bead_z_vals = np.zeros(self.n_atoms*self.n_samples)
        for i_sample in range(self.n_samples):
            for i_atom in range(self.n_atoms):
                bead_z_vals[i_sample*self.n_atoms + i_atom] = (self.atom_data[i_sample,i_atom].pos[2] * self.box_height)

        count,bins,x = plt.hist(bead_z_vals, bins=n_bins)

        bincenters = 0.5 * (bins[1:] + bins[:-1])
        plt.cla()
        plt.figure(figsize=(6,3))
        plt.plot(bincenters,(count*(sigma)**3)/(self.n_samples * (self.box_length**2 * (self.box_height/n_bins))),color='r')
        plt.ylim((0,0.8))
        plt.xlim((0,220))
        plt.xlabel('$z$ / $\AA$')
        plt.ylabel(r'$\rho(z)$ / $\sigma^{-3}$')
        plt.tight_layout()
        plt.savefig('results/264K.pgf')
        plt.show()

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

filename1 = 'md_data/264K.xyz'
filename2 = 'md_data/276K.xyz'
filename3 = 'md_data/288K.xyz'
filename4 = 'md_data/300K.xyz'

filename = 'md_data/264K_testdata.xyz'

ex6 = Exercise6(filename1)
print(ex6.n_samples)
ex6.compute_phase_diagram(40)