import numpy as np
import random
import base64
from matplotlib import pyplot as plt

from mpilammpswrapper import MpiLammpsWrapper

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


def generate_random_filename():
	return base64.b32encode(random.randbytes(5)).decode("ascii")


class MpiLammpsRun:
	def __init__(self, code: str, sim_params: dict, expect_dumps: list = None, file_name: str = None):
		self.output = ""
		self.code = code
		self.sim_params = sim_params
		self.file_name = f'/tmp/in.{generate_random_filename()}.lammps' if file_name is None else file_name
		self.expect_dumps = [] if expect_dumps is None else expect_dumps
		if 'cwd' in sim_params and sim_params['cwd'] is not None:
			self.expect_dumps = [f"{sim_params['cwd']}/{dump}" for dump in self.expect_dumps]
		else:
			print("WARN: No CWD!")
		self.dumps = []

	def execute(self) -> 'MpiLammpsRun':
		MpiLammpsWrapper.gen_and_sim(self.code, self.sim_params, file_to_use=self.file_name)
		self.dumps = self._parse_dumps()
		return self

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
		t = dump['atoms'][:, DUMP_ATOM_TYPE]
		x = dump['atoms'][:, DUMP_ATOM_X]
		y = dump['atoms'][:, DUMP_ATOM_Y]
		z = dump['atoms'][:, DUMP_ATOM_Z]
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
