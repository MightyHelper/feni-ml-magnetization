from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Generator

from model.live_execution import LiveExecution
from lammps.simulation_task import SimulationTask


@dataclass
class Machine(metaclass=ABCMeta):
    """
    A machine with a number of cores
    """
    name: str
    cores: int
    single_core_performance: float = field(init=False, default=1.0)

    task_queue: list[list[SimulationTask]] = field(init=False, default_factory=list)

    lammps_executable: PurePath
    execution_path: PurePath

    def __str__(self) -> str:
        return f"Machine: {self.name} ({self.cores} cores) with {sum([len(queue) for queue in self.task_queue])} tasks"

    def __repr__(self) -> str:
        return str(self)

    def __init__(self, name: str, cores: int, execution_path: PurePath, lammps_executable: PurePath):
        self.name = name
        self.cores = cores
        self.lammps_executable = lammps_executable
        self.execution_path = execution_path
        assert self.execution_path.is_absolute(), f"Execution path {self.execution_path} is not absolute"

    @abstractmethod
    def get_running_tasks(self) -> Generator[LiveExecution, None, None]:
        pass


