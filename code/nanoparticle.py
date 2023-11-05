import subprocess
import time
import os

import numpy as np

import mpilammpsrun as mpilr
import mpilammpswrapper as mpilw
import shapes
import template
import feni_mag
import feni_ovito

FE_ATOM = 1
NI_ATOM = 2


def realpath(path):
	return subprocess.check_output(["realpath", path]).decode("ascii").strip()


class Nanoparticle:
	"""Represents a nanoparticle"""
	regions: list[shapes.Shape]
	atom_manipulation: list[str]
	run: mpilr.MpiLammpsRun
	magnetism: tuple[float, float]

	def __init__(self, extra_replacements: dict = None):
		self.regions = []
		self.atom_manipulation = []
		self.extra_replacements = {} if extra_replacements is None else extra_replacements
		self.path = realpath("../executions/" + self.get_identifier()) + "/"
		print(f"Path: {self.path}")

	def add_shape(self, shape: shapes.Shape, action: str = 'create', atom_type: int = 1):
		self.regions.append(shape)
		region_index = len(self.regions) - 1
		if action == 'create':
			self.atom_manipulation.append(f"create_atoms {atom_type} region reg{region_index}")
		elif action == 'delete':
			self.atom_manipulation.append(f"delete_atoms {atom_type} region reg{region_index}")
		elif action == 'update':
			self.atom_manipulation.append(f"set region reg{region_index} type {atom_type}")
		return region_index

	def use_random_ratio(self, selector=("type", "1"), target_atom_type="2", ratio_to_convert="0.35", random_seed="250"):
		self.atom_manipulation.append(f"set {selector[0]} {selector[1]} type/ratio {target_atom_type} {ratio_to_convert} {random_seed}")

	def execute(self, test_run: bool = True, **kwargs):
		code = template.replace_templates(template.get_template(), {
			"region": self.get_region(),
			"run_steps": str(0 if test_run else 300000),
			**self.extra_replacements
		})
		os.mkdir(self.path)
		self.run = mpilr.MpiLammpsRun(
			code,
			{
				"lammps_executable": template.LAMMPS_EXECUTABLE,
				"omp": mpilw.OMPOpt(use=True, n_threads=2),
				"mpi": mpilw.MPIOpt(use=True, hw_threads=False, n_threads=4),
				"cwd": self.path,
				**kwargs
			},
			dumps := [f"iron.{i}.dump" for i in ([0] if test_run else [0, 100000, 200000, 300000])],
			file_name=self.path + "nanoparticle.in"
		).execute()
		feni_mag.extract_magnetism(self.path + "/log.lammps", out_mag=self.path + "/magnetism.txt", digits=4)
		self.magnetism = self.get_magnetism()
		feni_ovito.parse(filenames={'base_path': self.path, 'dump': dumps[-1]})

	def get_identifier(self):
		return f"simulation_{time.time()}"

	def get_magnetism(self):
		with open(self.path + "/magnetism.txt", "r") as f:
			lines = f.readlines()
		result = [float(x) for x in lines[1].split(" ")]
		return result[0], result[1]

	def get_region(self):
		out = ""
		for i in range(max(len(self.regions), len(self.atom_manipulation))):
			if i < len(self.regions):
				out += self.regions[i].get_region(f"reg{i}") + "\n"
			if i < len(self.atom_manipulation):
				out += self.atom_manipulation[i] + "\n"
		return out

	def plot(self):
		if self.run is None:
			raise Exception("Run not executed")

		self.run.plot(self.run.dumps[0])

	def count_atoms_of_type(self, atom_type, dump_idx=0):
		return np.count_nonzero(self.run.dumps[dump_idx]['atoms'][:, mpilr.DUMP_ATOM_TYPE] == atom_type)


class ParticleFactory:
	def __init__(self, base_volume: float):
		self.base_volume = base_volume

	@staticmethod
	def random_shape(shape: shapes.Shape, ratio: float = 0.35, random_seed: int = 250) -> Nanoparticle:
		nano = Nanoparticle()
		nano.add_shape(shape)
		nano.use_random_ratio(ratio_to_convert=str(ratio), random_seed=str(random_seed))
		return nano

	def random_sphere(self, ratio: float = 0.35, random_seed: int = 250) -> Nanoparticle:
		sphere = shapes.Sphere.from_volume(self.base_volume, (0, 0, 0))
		return ParticleFactory.random_shape(sphere, ratio, random_seed)

	def random_cylinder(self, radius: float = 10, ratio: float = 0.35, random_seed: int = 250) -> Nanoparticle:
		cylinder = shapes.Cylinder.from_volume_and_radius(self.base_volume, radius, 'x', (0, 0, 0))
		return ParticleFactory.random_shape(cylinder, ratio, random_seed)

	def core_shell_cylinder(self, outer_radius: float = 10, inner_radius: float = 10, ratio: float = 0.35) -> Nanoparticle:
		nano = Nanoparticle()
		outer_cylinder = shapes.Cylinder.from_volume_and_radius(self.base_volume, outer_radius, 'x', (0, 0, 0))
		inner_cylinder = shapes.Cylinder.from_volume_and_radius(self.base_volume * ratio, inner_radius, 'x', (0, 0, 0))
		print(f"{outer_cylinder=}\n{inner_cylinder=}")
		assert outer_cylinder.get_volume() > inner_cylinder.get_volume(), "Inner cylinder volume is greater than outer cylinder volume"
		assert outer_cylinder.radius > inner_cylinder.radius, "Inner cylinder radius is greater than outer cylinder radius"
		assert outer_cylinder.length > inner_cylinder.length, "Inner cylinder length is greater than outer cylinder length"
		assert outer_cylinder.get_volume() * ratio - inner_cylinder.get_volume() < 0.0001, "Volumes are not right"
		nano.add_shape(outer_cylinder, atom_type=FE_ATOM, action='create')
		nano.add_shape(inner_cylinder, atom_type=NI_ATOM, action='update')
		return nano
