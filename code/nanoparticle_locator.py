import os
from typing import Generator


class NanoparticleLocator:
	@staticmethod
	def search(path: str, extension: str = ".in") -> Generator[str, None, None]:
		"""
		Returns a generator of all files with the given extension in the given path
		:param path:  The path to search
		:param extension:  The extension to search for
		:return: A generator of all files with the given extension in the given path
		"""
		for file in os.listdir(path):
			if file.startswith("Test"):
				continue
			if os.path.isdir(f"{path}/{file}"):
				yield from NanoparticleLocator.search(f"{path}/{file}", extension)
			elif file.endswith(extension):
				yield f"{path}/{file}"

	@staticmethod
	def sorted_search(path: str, extension: str = ".in") -> Generator[str, None, None]:
		"""
		Returns a sorted list of all files with the given extension in the given path
		:param path:  The path to search
		:param extension:   The extension to search for
		:return:
		"""
		for file in sorted(os.listdir(path)):
			if file.startswith("Test"):
				continue
			if os.path.isdir(f"{path}/{file}"):
				yield from NanoparticleLocator.sorted_search(f"{path}/{file}", extension)
			elif file.endswith(extension):
				yield f"{path}/{file}"

	@staticmethod
	def get_a_particle(path: str = "../Shapes", extension: str = ".in") -> str:
		"""
		Returns the first particle found in the given path
		:param path:  The path to search
		:param extension:  The extension to search for
		:return:
		"""
		return next(iter(NanoparticleLocator.search(path, extension)))
