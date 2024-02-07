from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import AsyncGenerator

from model.live_execution import LiveExecution


@dataclass
class Machine(metaclass=ABCMeta):
    """
    A machine with a number of cores
    """
    name: str
    cores: int
    single_core_completion_time: float = field(init=False, default=60 * 17)  # Lower is better
    launch_time: float = field(init=False, default=0.0)  # Time to launch a task

    lammps_executable: PurePath
    execution_path: PurePath

    def __str__(self) -> str:
        return f"Machine: {self.name} ({self.cores} cores)"

    def __repr__(self) -> str:
        return str(self)

    def __init__(self, name: str, cores: int, execution_path: PurePath, lammps_executable: PurePath, launch_time: float = 0, single_core_completion_time: float = 1.0):
        self.name = name
        self.cores = cores
        self.lammps_executable = lammps_executable
        self.execution_path = execution_path
        self.launch_time = launch_time
        self.single_core_completion_time = single_core_completion_time
        assert self.execution_path.is_absolute(), f"Execution path {self.execution_path} is not absolute"

    @abstractmethod
    async def get_running_tasks(self) -> AsyncGenerator[LiveExecution, None]:
        pass
