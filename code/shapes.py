from numpy import pi
from dataclasses import dataclass
from abc import ABC, abstractmethod


class Shape(ABC):
	""" Represents a shape """

	@abstractmethod
	def get_volume(self) -> float:
		pass

	@abstractmethod
	def get_region(self, name: str) -> str:
		pass


@dataclass
class Sphere(Shape):
	""" Represents a sphere """
	radius: float
	center: tuple

	def get_volume(self) -> float:
		return 4.0 / 3.0 * pi * self.radius ** 3

	def get_region(self, name: str) -> str:
		return f"region {name} sphere {self.center[0]} {self.center[1]} {self.center[2]} {self.radius} units box"

	@staticmethod
	def from_volume(volume: float, center: tuple) -> 'Sphere':
		radius = (volume * 3 / (4 * pi)) ** (1 / 3)
		return Sphere(radius, center)

	@staticmethod
	def from_radius(radius: float, center: tuple) -> 'Sphere':
		return Sphere(radius, center)

	def __str__(self):
		return f"Sphere_{self.radius}_{self.center})"


@dataclass
class Cylinder(Shape):
	""" Represents a cylinder """
	radius: float
	length: float
	axis: str
	center: tuple

	def get_volume(self) -> float:
		return pi * self.radius ** 2 * self.length

	def get_region(self, name: str) -> str:
		a = 0
		b = 0
		c = 0
		if self.axis == 'x':
			a = self.center[1]
			b = self.center[2]
			c = self.center[0]
		if self.axis == 'y':
			a = self.center[0]
			b = self.center[2]
			c = self.center[1]
		if self.axis == 'z':
			a = self.center[0]
			b = self.center[1]
			c = self.center[2]
		return f"region {name} cylinder {self.axis} {a} {b} {self.radius} {c - self.length / 2.0} {c + self.length / 2.0} units box"

	@staticmethod
	def from_volume_and_radius(volume: float, radius: float, axis: str, center: tuple) -> 'Cylinder':
		length = volume / (pi * radius ** 2)
		return Cylinder(radius, length, axis, center)

	@staticmethod
	def from_volume_and_length(volume: float, length: float, axis: str, center: tuple) -> 'Cylinder':
		radius = (volume / (pi * length)) ** 0.5
		return Cylinder(radius, length, axis, center)

	@staticmethod
	def from_radius_and_length(radius: float, length: float, axis: str, center: tuple) -> 'Cylinder':
		return Cylinder(radius, length, axis, center)

	def __str__(self):
		return f"Cylinder_{self.radius}_{self.length}_{self.axis}_{self.center}"
