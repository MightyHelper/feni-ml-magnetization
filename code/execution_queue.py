import logging
import platform
import re
import subprocess
from abc import ABC, abstractmethod
from config import LAMMPS_EXECUTABLE
from simulation_task import SimulationTask


class ExecutionQueue(ABC):
	queue: list[SimulationTask] = []

	def enqueue(self, simulation_task: SimulationTask):
		assert isinstance(simulation_task, SimulationTask)
		assert simulation_task.input_file is not None
		assert simulation_task.cwd is not None
		assert simulation_task.mpi is not None
		assert simulation_task.omp is not None
		assert simulation_task.gpu is not None
		assert simulation_task not in self.queue
		self.queue.append(simulation_task)

	def _get_next_task(self):
		if len(self.queue) == 0:
			return None
		element = self.queue[0]
		self.queue = self.queue[1:]
		return element

	def run_callback(self, simulation_task: SimulationTask, result: str):
		for callback in simulation_task.callbacks:
			callback(result)

	def print_error(self, e, **kwargs):
		kwargs['queue'] = type(self)
		kwargs['queue'] = len(self.queue)
		params = "\n".join([f"{key}={value}" for key, value in kwargs.items()]) if kwargs else ""
		logging.error("ERROR:" + str(e) + params, extra={"markup": True})

	def run(self):
		while len(self.queue) > 0:
			try:
				self._simulate()
			except Exception as e:
				logging.error(f"Error in {type(self)}: {e}", stack_info=True)

	@abstractmethod
	def _simulate(self) -> None:
		pass


class LocalExecutionQueue(ExecutionQueue):
	def _simulate(self) -> None:
		simulation_task: SimulationTask = self._get_next_task()
		lammps_executable = LAMMPS_EXECUTABLE
		cmd = f"{simulation_task.mpi} {lammps_executable} {simulation_task.omp} {simulation_task.gpu} -in {simulation_task.input_file}"
		cmd = re.sub(r' +', " ", cmd).strip()
		logging.info(f"[bold blue]LocalExecutionQueue[/bold blue] Running [bold yellow]{cmd}[/bold yellow] in [cyan]{simulation_task.cwd}[/cyan]", extra={"markup": True, "highlighter": None})
		try:
			result = subprocess.check_output(cmd.split(" "), cwd=simulation_task.cwd, shell=platform.system() == "Windows")
			self.run_callback(simulation_task, result.decode("utf-8"))
		except subprocess.CalledProcessError as e:
			simulation_task.ok = False
			self.print_error(e)
			raise e
		except OSError as e:
			simulation_task.ok = False
			self.print_error(e)
			raise ValueError(f"Is LAMMPS ({lammps_executable}) installed?") from e
