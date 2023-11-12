import time
import os

import numpy as np

import mpilammpsrun as mpilr
import mpilammpswrapper as mpilw
import shapes
import template
import feni_mag
import feni_ovito
import random

FE_ATOM = 1
NI_ATOM = 2

FULL_RUN_DURATION = 300000


def realpath(path):
	return os.path.realpath(path)


class Nanoparticle:
	"""Represents a nanoparticle"""
	regions: list[shapes.Shape]
	atom_manipulation: list[str]
	run: mpilr.MpiLammpsRun
	region_name_map: dict[str, int] = {}
	magnetism: tuple[float, float]

	def __init__(self, extra_replacements: dict = None):
		self.regions = []
		self.atom_manipulation = []
		self.extra_replacements = {} if extra_replacements is None else extra_replacements
		self.rid = random.randint(0, 10000)
		self.title = self.extra_replacements.get("title", "Nanoparticle")
		self.path = realpath("../executions/" + self.get_identifier()) + "/"

	def execute(self, test_run: bool = True, **kwargs):
		"""
		Executes the nanoparticle simulation
		:param test_run:  If true, only one dump will be generated
		:param kwargs: 	Extra arguments to pass to the lammps run
		:return:
		"""
		code = self._build_lammps_code(test_run)
		os.mkdir(self.path)  # Create the run directory
		dumps, lammps_run = self._build_lammps_run(code, kwargs, test_run)
		self.run = lammps_run.execute()
		if not test_run:
			feni_mag.extract_magnetism(self.path + "/log.lammps", out_mag=self.path + "/magnetism.txt", digits=4)
			self.magnetism = self.get_magnetism()
			feni_ovito.parse(filenames={'base_path': self.path, 'dump': dumps[-1]})

	def _build_lammps_run(self, code, kwargs, test_run):
		lammps_run = mpilr.MpiLammpsRun(
			code, {
				"lammps_executable": template.LAMMPS_EXECUTABLE,
				"omp": mpilw.OMPOpt(use=False, n_threads=2),
				"mpi": mpilw.MPIOpt(use=False, hw_threads=False, n_threads=4),
				"cwd": self.path,
				**kwargs
			},
			dumps := [f"iron.{i}.dump" for i in ([0] if test_run else [*range(0, FULL_RUN_DURATION + 1, 100000)])],
			file_name=self.path + "nanoparticle.in"
		)
		return dumps, lammps_run

	def _build_lammps_code(self, test_run):
		return template.replace_templates(template.get_template(), {
			"region": self.get_region(),
			"run_steps": str(0 if test_run else FULL_RUN_DURATION),
			"title": f"{self.title}",
			**self.extra_replacements
		})

	def get_identifier(self):
		return f"simulation_{time.time()}_{self.rid}"

	def get_magnetism(self):
		with open(self.path + "/magnetism.txt", "r") as f:
			lines = f.readlines()
		result = [float(x) for x in lines[1].split(" ")]
		return result[0], result[1]

	def get_region(self):
		out = ""
		for i in self.atom_manipulation:
			if isinstance(i, str):
				out += i + "\n"
			elif isinstance(i, shapes.Shape):
				out += i.get_region(f"reg{self.regions.index(i)}") + "\n"
				raise Exception("Shapes are supported but not allowed")
			else:
				raise Exception(f"Unknown type: {type(i)}")
		return out

	def plot(self):
		if self.run is None:
			raise Exception("Run not executed")
		self.run.dumps[0].plot()

	def count_atoms_of_type(self, atom_type, dump_idx=0):
		return self.run.dumps[dump_idx].count_atoms_of_type(atom_type)

	def total_atoms(self, dump_idx=0):
		return self.run.dumps[dump_idx].dump['number_of_atoms']

	def __str__(self):
		return f"Nanoparticle(regions={len(self.regions)} items, atom_manipulation={len(self.atom_manipulation)} items, title={self.title})"

	def __repr__(self):
		return self.__str__()
