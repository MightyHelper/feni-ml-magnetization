from pathlib import Path
from unittest import TestCase

from lammps import nanoparticle_locator

PATH = Path("../Shapes")


class TestNanoparticleLocator(TestCase):
	def test_recursive_input_search(self):
		particles = list(nanoparticle_locator.NanoparticleLocator.search(PATH, extension=".in"))
		if len(particles) == 0:
			self.fail("No particles found")
		# particles = list(nanoparticle_locator.NanoparticleLocator.search(PATH, extension=".shrink"))
		# if len(particles) != 0:
		# 	self.fail("No shrinks particles found")
		particles = list(nanoparticle_locator.NanoparticleLocator.search(PATH, extension=".blabla"))
		if len(particles) != 0:
			self.fail("Found particles that don't exist")

	def test_sorted_search(self):
		particles = list(nanoparticle_locator.NanoparticleLocator.sorted_search(PATH, extension=".in"))
		if len(particles) == 0:
			self.fail("No particles found")
		self.assertListEqual(particles, sorted(particles))
		# particles = list(nanoparticle_locator.NanoparticleLocator.sorted_search(PATH, extension=".shrink"))
		# if len(particles) != 0:
		# 	self.fail("No shrinks particles found")
		# self.assertListEqual(particles, sorted(particles))
		particles = list(nanoparticle_locator.NanoparticleLocator.sorted_search(PATH, extension=".blabla"))
		if len(particles) != 0:
			self.fail("Found particles that don't exist")
