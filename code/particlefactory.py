import abc

import shapes
from nanoparticle import Nanoparticle, FE_ATOM, NI_ATOM
from nanoparticlebuilder import NanoparticleBuilder
import logging

class AbstractParticleFactory(abc.ABC):
	@staticmethod
	def random_shape(shape: shapes.Shape, ratio: float = 0.35, random_seed: int = 250) -> Nanoparticle:
		nano = NanoparticleBuilder()
		nano.add_shape(shape)
		nano.use_random_ratio(ratio_to_convert=str(ratio), random_seed=str(random_seed))
		return nano.build()

	@staticmethod
	def onion_cylinder(c1: tuple[float, float], c2: tuple[float, float], c3: tuple[float, float], c4: tuple[float, float]) -> Nanoparticle:
		nano = NanoparticleBuilder()
		size_4 = shapes.Cylinder(c1[0], c1[1], 'x', (0, 0, 0), True)
		size_3 = shapes.Cylinder(c2[0], c2[1], 'x', (0, 0, 0), True)
		size_2 = shapes.Cylinder(c3[0], c3[1], 'x', (0, 0, 0), True)
		size_1 = shapes.Cylinder(c4[0], c4[1], 'x', (0, 0, 0), True)
		nano.add_shape(size_4, atom_type=FE_ATOM, action='create')
		nano.add_shape(size_3, atom_type=NI_ATOM, action='update')
		nano.add_shape(size_2, atom_type=FE_ATOM, action='update')
		nano.add_shape(size_1, atom_type=NI_ATOM, action='update')
		return nano.build()

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
		nano = NanoparticleBuilder()
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
		return nano.build()


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
		nano = NanoparticleBuilder()
		inner_count = int(self.base_atom_count * ratio)
		outer_count = int(self.base_atom_count * (1 - ratio)) + inner_count
		logging.info(f"Outer count: {outer_count}, inner count: {inner_count}")
		outer_cylinder = shapes.Cylinder.from_lattice_point_count_and_radius(outer_count, outer_radius, 'x', (0, 0, 0))
		inner_cylinder = shapes.Cylinder.from_lattice_point_count_and_radius(inner_count, inner_radius, 'x', (0, 0, 0))
		logging.info(f"Outer: {outer_cylinder}")
		logging.info(f"Inner: {inner_cylinder}")
		ilc = inner_cylinder.get_lattice_point_count()
		olc = outer_cylinder.get_lattice_point_count() - ilc
		assert olc > ilc, "Inner cylinder volume is greater than outer cylinder volume"
		assert outer_cylinder.radius > inner_cylinder.radius, "Inner cylinder radius is greater than outer cylinder radius"
		assert outer_cylinder.length > inner_cylinder.length, "Inner cylinder length is greater than outer cylinder length"
		assert (olc + ilc) - self.base_atom_count < 50, f"Volumes are not right ({outer_cylinder.get_lattice_point_count()} {inner_cylinder.get_lattice_point_count()}) {(olc + ilc) - self.base_atom_count}"
		assert ilc / (olc + ilc) - ratio < 0.1, f"Volumes are not right ({(olc + ilc) / ilc}!={ratio})"
		nano.add_shape(outer_cylinder, atom_type=FE_ATOM, action='create')
		nano.add_shape(inner_cylinder, atom_type=NI_ATOM, action='update')
		return nano.build()
