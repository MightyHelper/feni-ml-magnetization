import os

import numpy as np
from matplotlib import pyplot as plt

from utils import get_index, opt, column_values_as_float


class LammpsDump:
    """
    Functions to parse a dump file
    """

    def __init__(self, path):
        self.path = path
        self.dump = self._parse()

    def _parse(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dump file {self.path} does not exist!")
        with open(self.path, "r") as f:
            lines = f.readlines()
            return {
                "timestep": opt(
                    get_index(lines, "TIMESTEP", "ITEM: "),
                    lambda x: int(lines[x + 1])
                ),
                "number_of_atoms": opt(
                    get_index(lines, "NUMBER OF ATOMS", "ITEM: "),
                    lambda x: int(lines[x + 1])
                ),
                "box_bounds": opt(
                    get_index(lines, "BOX BOUNDS", "ITEM: "),
                    lambda o: np.array([
                        column_values_as_float(line)
                        for line in lines[o + 1:o + 4]
                    ])
                ),
                "atoms": opt(
                    get_index(lines, "ATOMS", "ITEM: "),
                    lambda o: np.array([
                        column_values_as_float(line)
                        for line in lines[o + 1:]
                    ])
                )
            }

    def plot(self):
        t = self.dump['atoms'][:, DUMP_ATOM_TYPE]
        x = self.dump['atoms'][:, DUMP_ATOM_X]
        y = self.dump['atoms'][:, DUMP_ATOM_Y]
        z = self.dump['atoms'][:, DUMP_ATOM_Z]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=t, marker='.', cmap='coolwarm')
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(self.dump['box_bounds'][0][0], self.dump['box_bounds'][0][1])
        ax.set_ylim(self.dump['box_bounds'][1][0], self.dump['box_bounds'][1][1])
        ax.set_zlim(self.dump['box_bounds'][2][0], self.dump['box_bounds'][2][1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def count_atoms_of_type(self, atom_type):
        return np.count_nonzero(self.dump['atoms'][:, DUMP_ATOM_TYPE] == atom_type)

    def __str__(self):
        return f"<Dump@{self.dump['timestep']} for {self.path}>"

    def __repr__(self):
        return str(self)


DUMP_ATOM_TYPE = 0
DUMP_ATOM_ID = 1
DUMP_ATOM_X = 2
DUMP_ATOM_Y = 3
DUMP_ATOM_Z = 4
DUMP_ATOM_VX = 5
DUMP_ATOM_VY = 6
DUMP_ATOM_VZ = 7
DUMP_ATOM_C1 = 8
DUMP_ATOM_C2 = 9
DUMP_ATOM_C3 = 10
DUMP_ATOM_PE = 11
DUMP_ATOM_KE = 12
