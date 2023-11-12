import abc
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

	# print(f"Path: {self.path}")

	def add_shape(self, shape: shapes.Shape, action: str = 'create', atom_type: int = 1):
		region_index = len(self.regions)
		self.regions.append(shape)
		self.atom_manipulation.append(shape.get_region(f"reg{region_index}"))
		if action == 'create':
			self.add_create_atoms(atom_type, region_index)
		elif action == 'delete':
			self.atom_manipulation.append(f"delete_atoms {atom_type} region reg{region_index}")
		elif action == 'update':
			self.atom_manipulation.append(f"set region reg{region_index} type {atom_type}")
		else:
			raise Exception(f"Unknown action: {action}")
		return region_index

	def use_random_ratio(self, selector=("type", "1"), target_atom_type="2", ratio_to_convert="0.35", random_seed="250"):
		self.atom_manipulation.append(f"set {selector[0]} {selector[1]} type/ratio {target_atom_type} {ratio_to_convert} {random_seed}")

	def execute(self, test_run: bool = True, **kwargs):
		code = template.replace_templates(template.get_template(), {
			"region": self.get_region(),
			"run_steps": str(0 if test_run else FULL_RUN_DURATION),
			"title": f"{self.title}",
			**self.extra_replacements
		})
		os.mkdir(self.path)
		self.run = mpilr.MpiLammpsRun(
			code,
			{
				"lammps_executable": template.LAMMPS_EXECUTABLE,
				"omp": mpilw.OMPOpt(use=False, n_threads=2),
				"mpi": mpilw.MPIOpt(use=False, hw_threads=False, n_threads=4),
				"cwd": self.path,
				**kwargs
			},
			dumps := [f"iron.{i}.dump" for i in ([0] if test_run else [*range(0, FULL_RUN_DURATION + 1, 100000)])],
			file_name=self.path + "nanoparticle.in"
		).execute()
		if not test_run:
			feni_mag.extract_magnetism(self.path + "/log.lammps", out_mag=self.path + "/magnetism.txt", digits=4)
			self.magnetism = self.get_magnetism()
			feni_ovito.parse(filenames={'base_path': self.path, 'dump': dumps[-1]})

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
		return np.count_nonzero(self.run.dumps[dump_idx].dump['atoms'][:, mpilr.DUMP_ATOM_TYPE] == atom_type)

	def total_atoms(self, dump_idx=0):
		return self.run.dumps[dump_idx].dump['number_of_atoms']

	def add_named_shape(self, shape, region_name):
		region_id = self.register_region_name(region_name)
		self.regions.append(shape)
		self.atom_manipulation.append(shape.get_region(f"reg{region_id}"))

	def register_region_name(self, region_name):
		self.region_name_map[region_name] = len(self.regions)
		return self.region_name_map[region_name]

	def get_region_id_by_name(self, region_name):
		return self.region_name_map[region_name]

	def add_create_atoms(self, atom_type, region_name):
		if isinstance(region_name, str):
			region_name = self.get_region_id_by_name(region_name)
		region_name = int(region_name)
		command = f"create_atoms {atom_type} region reg{region_name}"
		self.atom_manipulation.append(command)
		return command

	def add_set_type_region(self, atom_type, region_name):
		if isinstance(region_name, str):
			region_name = self.get_region_id_by_name(region_name)
		region_name = int(region_name)
		command = f"set region reg{region_name} type {atom_type}"
		self.atom_manipulation.append(command)
		return command

	def add_set_type_subset_region(self, atom_type, region_name, count, seed):
		if isinstance(region_name, str):
			region_name = self.get_region_id_by_name(region_name)
		region_name = int(region_name)
		command = f"set region reg{region_name} type/subset {atom_type} {count} {seed}"
		self.atom_manipulation.append(command)
		return command

	def add_set_type_group(self, atom_type, group_name):
		command = f"set group {group_name} type {atom_type}"
		self.atom_manipulation.append(command)
		return command

	def add_set_type_subset_group(self, atom_type, group_name, count, seed):
		command = f"set group {group_name} type/subset {atom_type} {count} {seed}"
		self.atom_manipulation.append(command)
		return command

	def add_set_type_ratio_group(self, atom_type, group_name, ratio, seed):
		command = f"set group {group_name} type/ratio {atom_type} {ratio} {seed}"
		self.atom_manipulation.append(command)
		return command

	def add_group_type(self, atom_type, group_name):
		command = f"group {group_name} type {atom_type}"
		self.atom_manipulation.append(command)
		return command

	def __str__(self):
		return f"Nanoparticle(regions={len(self.regions)}, atom_manipulation={len(self.atom_manipulation)}, title={self.title})"

	def __repr__(self):
		return self.__str__()

	def add_intersect(self, regions: list[str], region_name: str):
		region_indices = [self.get_region_id_by_name(region) for region in regions]
		region_indices = [f"reg{i}" for i in region_indices]
		new_region_id = self.register_region_name(region_name)
		command = f"region reg{new_region_id} intersect {len(regions)} {' '.join(region_indices)} units box"
		self.atom_manipulation.append(command)
		return command


class AbstractParticleFactory(abc.ABC):
	@staticmethod
	def random_shape(shape: shapes.Shape, ratio: float = 0.35, random_seed: int = 250) -> Nanoparticle:
		nano = Nanoparticle()
		nano.add_shape(shape)
		nano.use_random_ratio(ratio_to_convert=str(ratio), random_seed=str(random_seed))
		return nano

	@staticmethod
	def onion_cylinder(c1, c2, c3, c4):
		# onion_cylinder((12.5, 30), (9, 26), (6, 22), (3, 18))
		nano = Nanoparticle()
		size_4 = shapes.Cylinder(c1[0], c1[1], 'x', (0, 0, 0), True)
		size_3 = shapes.Cylinder(c2[0], c2[1], 'x', (0, 0, 0), True)
		size_2 = shapes.Cylinder(c3[0], c3[1], 'x', (0, 0, 0), True)
		size_1 = shapes.Cylinder(c4[0], c4[1], 'x', (0, 0, 0), True)
		nano.add_shape(size_4, atom_type=FE_ATOM, action='create')
		nano.add_shape(size_3, atom_type=NI_ATOM, action='update')
		nano.add_shape(size_2, atom_type=FE_ATOM, action='update')
		nano.add_shape(size_1, atom_type=NI_ATOM, action='update')
		return nano

	@staticmethod
	def by_volume(base_volume: float) -> 'AbstractParticleFactory':
		return VolumeParticleFactory(base_volume)

	@staticmethod
	def by_atom_count(base_atom_count: float) -> 'AbstractParticleFactory':
		return AtomParticleFactory(base_atom_count)

	@abc.abstractmethod
	def random_sphere(self, ratio: float = 0.35, random_seed: int = 250) -> Nanoparticle:
		pass

	@abc.abstractmethod
	def random_cylinder(self, radius: float = 10, ratio: float = 0.35, random_seed: int = 250) -> Nanoparticle:
		pass

	@abc.abstractmethod
	def core_shell_cylinder(self, outer_radius: float = 10, inner_radius: float = 10, ratio: float = 0.35) -> Nanoparticle:
		pass


class VolumeParticleFactory(AbstractParticleFactory):
	def __init__(self, base_volume: float):
		self.base_volume = base_volume

	def random_sphere(self, ratio: float = 0.35, random_seed: int = 250) -> Nanoparticle:
		sphere = shapes.Sphere.from_volume(self.base_volume, (0, 0, 0))
		return AbstractParticleFactory.random_shape(sphere, ratio, random_seed)

	def random_cylinder(self, radius: float = 10, ratio: float = 0.35, random_seed: int = 250) -> Nanoparticle:
		cylinder = shapes.Cylinder.from_volume_and_radius(self.base_volume, radius, 'x', (0, 0, 0))
		return AbstractParticleFactory.random_shape(cylinder, ratio, random_seed)

	def core_shell_cylinder(self, outer_radius: float = 10, inner_radius: float = 10, ratio: float = 0.35) -> Nanoparticle:
		nano = Nanoparticle()
		outer_volume = self.base_volume * (1 - ratio)
		inner_volume = self.base_volume * ratio
		outer_cylinder = shapes.Cylinder.from_volume_and_radius(inner_volume, outer_radius, 'x', (0, 0, 0))
		inner_cylinder = shapes.Cylinder.from_volume_and_radius(outer_volume, inner_radius, 'x', (0, 0, 0))
		assert outer_cylinder.get_volume() > inner_cylinder.get_volume(), "Inner cylinder volume is greater than outer cylinder volume"
		assert outer_cylinder.radius > inner_cylinder.radius, "Inner cylinder radius is greater than outer cylinder radius"
		assert outer_cylinder.length > inner_cylinder.length, "Inner cylinder length is greater than outer cylinder length"
		assert outer_cylinder.get_volume() * ratio - inner_cylinder.get_volume() < 0.0001, "Volumes are not right"
		nano.add_shape(outer_cylinder, atom_type=FE_ATOM, action='create')
		nano.add_shape(inner_cylinder, atom_type=NI_ATOM, action='update')
		return nano


class AtomParticleFactory(AbstractParticleFactory):
	def __init__(self, base_atom_count: float):
		self.base_atom_count = base_atom_count

	def random_sphere(self, ratio: float = 0.35, random_seed: int = 250) -> Nanoparticle:
		sphere = shapes.Sphere.from_lattice_point_count(int(self.base_atom_count), (0, 0, 0))
		return AbstractParticleFactory.random_shape(sphere, ratio, random_seed)

	def random_cylinder(self, radius: float = 10, ratio: float = 0.35, random_seed: int = 250) -> Nanoparticle:
		cylinder = shapes.Cylinder.from_lattice_point_count_and_radius(int(self.base_atom_count), radius, 'x', (0, 0, 0))
		return AbstractParticleFactory.random_shape(cylinder, ratio, random_seed)

	def core_shell_cylinder(self, outer_radius: float = 10, inner_radius: float = 10, ratio: float = 0.35) -> Nanoparticle:
		nano = Nanoparticle()
		inner_count = int(self.base_atom_count * ratio)
		outer_count = int(self.base_atom_count * (1 - ratio)) + inner_count
		print(f"Outer count: {outer_count}, inner count: {inner_count}")
		outer_cylinder = shapes.Cylinder.from_lattice_point_count_and_radius(outer_count, outer_radius, 'x', (0, 0, 0))
		inner_cylinder = shapes.Cylinder.from_lattice_point_count_and_radius(inner_count, inner_radius, 'x', (0, 0, 0))
		print(f"Outer: {outer_cylinder}")
		print(f"Inner: {inner_cylinder}")
		ilc = inner_cylinder.get_lattice_point_count()
		olc = outer_cylinder.get_lattice_point_count() - ilc
		assert olc > ilc, "Inner cylinder volume is greater than outer cylinder volume"
		assert outer_cylinder.radius > inner_cylinder.radius, "Inner cylinder radius is greater than outer cylinder radius"
		assert outer_cylinder.length > inner_cylinder.length, "Inner cylinder length is greater than outer cylinder length"
		assert (olc + ilc) - self.base_atom_count < 50, f"Volumes are not right ({outer_cylinder.get_lattice_point_count()} {inner_cylinder.get_lattice_point_count()}) {(olc + ilc) - self.base_atom_count}"
		assert ilc / (olc + ilc) - ratio < 0.1, f"Volumes are not right ({(olc + ilc) / ilc}!={ratio})"
		nano.add_shape(outer_cylinder, atom_type=FE_ATOM, action='create')
		nano.add_shape(inner_cylinder, atom_type=NI_ATOM, action='update')
		return nano
