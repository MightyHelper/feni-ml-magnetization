import logging
import os
import re

import numpy as np
import random
import base64
from matplotlib import pyplot as plt

import config
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


def get_index(lines, section):
	return [i for i, l in enumerate(lines) if l.startswith("ITEM: " + section)][0]


class MPILammpsDump:
	"""
	Functions to parse a dump file
	"""
	def __init__(self, path):
		self.path = path
		self.dump = self._parse()

	def _parse(self):
		with open(self.path, "r") as f:
			lines = f.readlines()
			timestep_index = get_index(lines, "TIMESTEP")
			number_of_atoms_index = get_index(lines, "NUMBER OF ATOMS")
			box_bounds_index = get_index(lines, "BOX BOUNDS")
			atoms_index = get_index(lines, "ATOMS")
			timestep = int(lines[timestep_index + 1])
			number_of_atoms = int(lines[number_of_atoms_index + 1])
			box_bounds = np.array([[float(x) for x in line.split(" ") if x != ""] for line in lines[box_bounds_index + 1:box_bounds_index + 4]])
			atoms = np.array([[float(x) for x in line.split(" ") if x != ""] for line in lines[atoms_index + 1:]])
			return {
				"timestep": timestep,
				"number_of_atoms": number_of_atoms,
				"box_bounds": box_bounds,
				"atoms": atoms
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


class MpiLammpsRun:
	"""
	Functions to execute a lammps run
	"""
	def __init__(self, code: str, sim_params: dict, expect_dumps: list = None, file_name: str = None):
		self.output = ""
		self.code = code
		self.sim_params = sim_params
		self.file_name = f'/tmp/in.{generate_random_filename()}.lammps' if file_name is None else file_name
		self.expect_dumps = [] if expect_dumps is None else expect_dumps
		if 'cwd' in sim_params and sim_params['cwd'] is not None:
			self.expect_dumps = [f"{sim_params['cwd']}/{dump}" for dump in self.expect_dumps]
		else:
			logging.warning("No CWD passed to sim_params!")
		self.dumps: list[MPILammpsDump] = []

	def get_lammps_log_filename(self):
		return self.sim_params['cwd'] + "/log.lammps"

	def get_current_step(self):
		"""
		Get the current step of a lammps log file
		"""
		step = -1
		try:
			with open(self.get_lammps_log_filename(), "r") as f:
				lines = f.readlines()
				try:
					split = re.split(r" +", lines[-1].strip())
					step = int(split[0])
				except Exception:
					pass
		except FileNotFoundError:
			pass
		return step

	def execute(self) -> 'MpiLammpsRun':
		MpiLammpsWrapper.gen_and_sim(self.code, self.sim_params, file_to_use=self.file_name)
		self.dumps = self._parse_dumps()
		return self

	def _parse_dumps(self):
		dumps = {}
		for dump in self.expect_dumps:
			result = MPILammpsDump(dump)
			dumps[result.dump['timestep']] = result
		return dumps


	@staticmethod
	def from_path(path):
		files = os.listdir(path)
		nano_in = [path + "/" + file for file in files if file.endswith(".in")][0]
		with open(nano_in, "r") as f:
			code = f.read()
		dumps = [file for file in files if file.endswith(".dump")]
		lr = MpiLammpsRun(code, {'cwd': path}, dumps, nano_in)
		lr.dumps = lr._parse_dumps()
		return lr
