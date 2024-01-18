import itertools
import os
from pathlib import Path
from typing import Generator


class NanoparticleLocator:
	@staticmethod
	def search(path: Path, extension: str = ".in") -> Generator[Path, None, None]:
		"""
		Returns a generator of all files with the given extension in the given path
		:param path:  The path to search
		:param extension:  The extension to search for
		:return: A generator of all files with the given extension in the given path
		"""
		for file in os.listdir(path):
			if file.startswith("Test"):
				continue
			if os.path.isdir(path / file):
				yield from NanoparticleLocator.search(path / file, extension)
			elif file.endswith(extension):
				yield path / file

	@staticmethod
	def sorted_search(path: Path, extension: str = ".in") -> Generator[Path, None, None]:
		"""
		Returns a sorted list of all files with the given extension in the given path
		:param path:  The path to search
		:param extension:   The extension to search for
		:return:
		"""
		for file in sorted(os.listdir(path)):
			if file.startswith("Test"):
				continue
			if os.path.isdir(path / file):
				yield from NanoparticleLocator.sorted_search(path / file, extension)
			elif file.endswith(extension):
				yield path / file

	@staticmethod
	def get_a_particle(path: Path = Path("../Shapes"), extension: str = ".in", index: int = 0) -> Path:
		"""
		Returns the first particle found in the given path
		:param path:  The path to search
		:param extension:  The extension to search for
		:param index:  The index of the particle to return
		:return:
		"""
		generator = NanoparticleLocator.search(path, extension)
		return next(itertools.islice(generator, index, None))
