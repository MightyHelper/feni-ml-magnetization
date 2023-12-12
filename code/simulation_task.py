import dataclasses
from dataclasses import field
from typing import Callable

from opt import GPUOpt, MPIOpt, OMPOpt


@dataclasses.dataclass
class SimulationTask:
	input_file: str = field(default_factory=str)
	gpu: GPUOpt = field(default_factory=GPUOpt)
	mpi: MPIOpt = field(default_factory=MPIOpt)
	omp: OMPOpt = field(default_factory=OMPOpt)
	cwd: str = field(default_factory=str)
	is_test_run: bool = field(default_factory=lambda: False)
	callbacks: list[Callable[[str], None]] = field(default_factory=list)

	def add_callback(self, callback: Callable[[str], None]):
		self.callbacks.append(callback)
