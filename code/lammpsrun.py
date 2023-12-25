import logging
import os
import re

from lammpsdump import LammpsDump
from simulationwrapper import SimulationWrapper
from simulation_task import SimulationTask
from utils import generate_random_filename


class LammpsRun:
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
		self.dumps: dict[int, LammpsDump] = {}

	def get_lammps_log_filename(self):
		return self.sim_params['cwd'] + "/log.lammps"

	@staticmethod
	def compute_current_step(lammps_log_contents: str):
		"""
		Get the current step of a lammps log file
		"""
		step = -1
		lines = [line.strip() for line in lammps_log_contents.split("\n") if line.strip() != ""]
		# noinspection PyBroadException
		try:
			split = re.split(r" +", lines[-1])
			step = int(split[0])
		except Exception:
			logging.debug(f"Could not parse step from {lines}")
			pass
		return step

	def get_current_step(self):
		"""
		Get the current step of a lammps log file
		"""
		try:
			with open(self.get_lammps_log_filename(), "r") as f:
				return LammpsRun.compute_current_step(f.read())
		except FileNotFoundError:
			pass
		return -1

	def get_simulation_task(self) -> SimulationTask:
		"""
		Get a simulation task for this run
		"""
		sim_task = SimulationWrapper.generate(self.code, self.sim_params, file_to_use=self.file_name)
		sim_task.add_callback(self.on_post_execution)
		return sim_task

	def on_post_execution(self, result: str) -> None:
		self.output = result
		self.dumps = self._parse_dumps()
		logging.debug(f"Finished execution of {self.file_name}")

	def _parse_dumps(self):
		dumps = {}
		for dump in self.expect_dumps:
			logging.debug(f"Parsing dump {dump}")
			result = LammpsDump(dump)
			logging.debug("Result: " + str(result))
			dumps[result.dump['timestep']] = result
		return dumps

	@staticmethod
	def from_path(path):
		files = os.listdir(path)
		nano_in = [path + "/" + file for file in files if file.endswith(".in")][0]
		with open(nano_in, "r") as f:
			code = f.read()
		dumps = [file for file in files if file.endswith(".dump")]
		lr = LammpsRun(code, {'cwd': path}, dumps, nano_in)
		lr.dumps = lr._parse_dumps()
		return lr
