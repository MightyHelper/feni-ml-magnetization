import logging
import os
import re
from pathlib import Path
from typing import Any

import utils
from lammpsdump import LammpsDump
from simulation_task import SimulationTask, SimulationWrapper
from utils import generate_random_filename


class LammpsRun:
	"""
	Functions to execute a lammps run
	"""
	output: str
	code: str
	sim_params: dict[str, Any]
	file_name: Path
	cwd: Path
	expect_dumps: list[Path]
	dumps: dict[int, LammpsDump]

	def __init__(self, code: str, sim_params: dict, expect_dumps: list = None, file_name: Path = None):
		self.output = ""
		self.code = code
		self.sim_params = sim_params
		self.file_name = Path(f'/tmp/in.{generate_random_filename()}.lammps') if file_name is None else file_name
		self.expect_dumps = [] if expect_dumps is None else expect_dumps
		self.cwd = Path(sim_params['cwd']).resolve() if 'cwd' in sim_params and sim_params['cwd'] is not None else Path.cwd().resolve()
		self.expect_dumps = [self.cwd / dump for dump in self.expect_dumps]
		self.dumps: dict[int, LammpsDump] = {}

	def get_lammps_log_filename(self) -> Path:
		return self.cwd / "log.lammps"

	@staticmethod
	def compute_current_step(lammps_log_contents: str) -> int:
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

	def get_current_step(self) -> int:
		"""
		Get the current step of a lammps log file
		"""
		try:
			LammpsRun.compute_current_step(utils.read_local_file(self.get_lammps_log_filename()))
		except FileNotFoundError:
			pass
		return -1

	def get_simulation_task(self) -> SimulationTask:
		"""
		Get a simulation task for this run
		"""
		sim_task = SimulationWrapper.generate(self.code, file_to_use=self.file_name, sim_params=self.sim_params)
		sim_task.add_callback(self.on_post_execution)
		return sim_task

	def on_post_execution(self, result: str | None) -> None:
		if result is None:
			logging.debug(f"Execution failed for lammps run {self.file_name}")
			return
		self.output = result
		self.dumps = self._parse_dumps()
		logging.debug(f"Finished execution of {self.file_name}")
		logging.debug(f"Lammps run {self} done with {len(self.dumps)} dumps")

	def _parse_dumps(self):
		dumps = {}
		for dump in self.expect_dumps:
			logging.debug(f"Parsing dump {dump}")
			result = LammpsDump(dump)
			logging.debug("Result: " + str(result))
			dumps[result.dump['timestep']] = result
		return dumps

	@staticmethod
	def from_path(path: Path):
		files: list[str] = os.listdir(path)
		nano_in: Path = [path / file for file in files if file.endswith(".in")][0]
		code: str = utils.read_local_file(nano_in)
		dumps: list[Path] = [path / file for file in files if file.endswith(".dump")]
		lr = LammpsRun(code, {'cwd': path}, dumps, nano_in)
		lr.dumps = lr._parse_dumps()
		return lr
