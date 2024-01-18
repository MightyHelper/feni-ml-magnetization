import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from pathlib import PurePath, PurePosixPath
from typing import Callable, Generator

from model.live_execution import LiveExecution
from simulation_task import SimulationTask


@dataclass
class Machine(metaclass=ABCMeta):
    """
    A machine with a number of cores
    """
    name: str
    cores: int
    single_core_performance: float = field(init=False, default=1.0)

    task_queue: list[list[SimulationTask]] = field(init=False, default_factory=list)

    lammps_executable: PurePosixPath
    execution_path: PurePosixPath

    def __str__(self) -> str:
        return f"Machine: {self.name} ({self.cores} cores) with {sum([len(queue) for queue in self.task_queue])} tasks"

    def __repr__(self) -> str:
        return str(self)

    def __init__(self, name: str, cores: int, execution_path: PurePosixPath, lammps_executable: PurePosixPath):
        self.name = name
        self.cores = cores
        self.lammps_executable = lammps_executable
        self.execution_path = execution_path

    @abstractmethod
    def run_cmd(self, command_getter: Callable[[str, str], list[str]]) -> bytes:
        pass

    @abstractmethod
    def mkdir(self, remote_path: str):
        pass

    @abstractmethod
    def cp_to(self, local_path: str, remote_path: str, is_folder: bool):
        pass

    @abstractmethod
    def cp_multi_to(self, local_paths: list[str], remote_path: str):
        pass

    @abstractmethod
    def cp_multi_from(self, remote_paths: list[str], local_path: str):
        pass

    @abstractmethod
    def cp_from(self, remote_path: str, local_path: str, is_folder: bool):
        pass

    @abstractmethod
    def read_file(self, filename: str) -> str:
        pass

    @abstractmethod
    def read_multiple_files(self, filenames: list[str]) -> list[str]:
        pass

    @abstractmethod
    def rm(self, file_path: str):
        pass

    @abstractmethod
    def ls(self, remote_dir: str) -> list[str]:
        pass

    @abstractmethod
    def remove_dir(self, remote_dir: str):
        pass

    def copy_alloy_files(self, local_sim_folder: PurePath, remote_sim_folder: PurePath):
        local_alloy_file: PurePath = local_sim_folder.parent.parent / "FeCuNi.eam.alloy"
        remote_alloy_file: PurePath = remote_sim_folder.parent.parent / "FeCuNi.eam.alloy"
        logging.info("Copying alloy files...")
        self.cp_to(local_alloy_file.as_posix(), remote_alloy_file.as_posix(), False)

    @abstractmethod
    def get_running_tasks(self) -> Generator[LiveExecution, None, None]:
        pass


