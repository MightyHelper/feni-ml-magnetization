import logging
import platform
import re
import subprocess
import toko_utils as tokoutil
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
		kwargs['queue'] = self.queue
		kwargs['queue_obj'] = self
		params = "\n".join([f"{key}={value}" for key, value in kwargs.items()]) if kwargs else ""
		logging.error("ERROR:" + str(e) + params, extra={"markup": True})

	def run(self):
		while len(self.queue) > 0:
			self._simulate()

	@abstractmethod
	def _simulate(self) -> None:
		pass


class LocalExecutionQueue(ExecutionQueue):
	def _simulate(self) -> None:
		simulation_task: SimulationTask = self._get_next_task()
		lammps_executable = LAMMPS_EXECUTABLE
		cmd = f"{simulation_task.mpi} {lammps_executable} {simulation_task.omp} {simulation_task.gpu} -in {simulation_task.input_file}"
		cmd = re.sub(r' +', " ", cmd).strip()
		try:
			logging.info(f"[bold blue]LocalExecutionQueue[/bold blue] Running [bold yellow]{cmd}[/bold yellow] in [cyan]{simulation_task.cwd}[/cyan]", extra={"markup": True, "highlighter": None})
			result = subprocess.check_output(cmd.split(" "), cwd=simulation_task.cwd, shell=platform.system() == "Windows")
			self.run_callback(simulation_task, result.decode("utf-8"))
		except subprocess.CalledProcessError as e:
			self.print_error(e)
			raise e
		except OSError as e:
			self.print_error(e)
			raise ValueError(f"Is LAMMPS ({lammps_executable}) installed?") from e


class TokoExecutionQueue(ExecutionQueue):
	def _simulate(self) -> None:
		simulation_task: SimulationTask = self._get_next_task()
		try:
			logging.info(f"[bold green]TokoExecutionQueue[/bold green] Running [bold yellow]{simulation_task.input_file}[/bold yellow] in [cyan]{simulation_task.cwd}[/cyan]", extra={"markup": True, "highlighter": None})
			result = tokoutil.simulate_in_toko(simulation_task.input_file)
			simulation_task.callback(result)
		except subprocess.CalledProcessError as e:
			self.print_error(e)
			raise e
		except OSError as e:
			self.print_error(e)
			raise ValueError(f"Is LAMMPS ({LAMMPS_EXECUTABLE}) installed?") from e


class TokoBatchedExecutionQueue(ExecutionQueue):
	def __init__(self, batch_size: int = 10):
		super().__init__()
		self.batch_size = batch_size

	def _simulate(self):
		simulations = self._get_next_task()
		logging.warning(f"[red]\[Not implemented] Running {len(simulations)} simulations in toko[/red]")

	def _get_next_task(self):
		if len(self.queue) == 0:
			return None
		next_task = self.queue[:self.batch_size]
		self.queue = self.queue[self.batch_size:]
		return next_task
