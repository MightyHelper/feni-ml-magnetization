import numpy as np
import random
import base64
from matplotlib import pyplot as plt

from mpilammpswrapper import MpiLammpsWrapper


def generate_random_filename():
	return base64.b32encode(random.randbytes(5)).decode("ascii")


class MpiLammpsRun:
	def __init__(self, code: str, sim_params: dict, expect_dumps: list = None):
		self.output = MpiLammpsWrapper.gen_and_sim(code, sim_params, file_to_use=f'/tmp/in.{generate_random_filename()}.lammps')
		self.expect_dumps = [] if expect_dumps is None else expect_dumps
		self.dumps = self._parse_dumps()

	def _parse_dumps(self):
		dumps = {}
		for dump in self.expect_dumps:
			result = self._parse_dump(dump)
			dumps[result['timestep']] = result
		return dumps

	def _parse_dump(self, dump):
		with open(dump, "r") as f:
			lines = f.readlines()
			timestep_index = self.get_index(lines, "TIMESTEP")
			number_of_atoms_index = self.get_index(lines, "NUMBER OF ATOMS")
			box_bounds_index = self.get_index(lines, "BOX BOUNDS")
			atoms_index = self.get_index(lines, "ATOMS")
			timestep = int(lines[timestep_index + 1])
			number_of_atoms = int(lines[number_of_atoms_index + 1])
			box_bounds = np.array([[float(x) for x in line.split(" ") if x != ""] for line in lines[box_bounds_index + 1:box_bounds_index + 4]])
			atoms = np.array([[float(x) for x in line.split(" ") if x != ""] for line in lines[atoms_index + 1:-1]])
			return {
				"timestep": timestep,
				"number_of_atoms": number_of_atoms,
				"box_bounds": box_bounds,
				"atoms": atoms
			}

	def plot(self, dump):
		t = dump['atoms'][:, 0]
		x = dump['atoms'][:, 2]
		y = dump['atoms'][:, 3]
		z = dump['atoms'][:, 4]
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(x, y, z, c=t, marker='.', cmap='coolwarm')
		ax.set_box_aspect((1, 1, 1))
		ax.set_xlim(dump['box_bounds'][0][0], dump['box_bounds'][0][1])
		ax.set_ylim(dump['box_bounds'][1][0], dump['box_bounds'][1][1])
		ax.set_zlim(dump['box_bounds'][2][0], dump['box_bounds'][2][1])
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		plt.show()

	def get_index(self, lines, section):
		return [i for i, l in enumerate(lines) if l.startswith("ITEM: " + section)][0]
