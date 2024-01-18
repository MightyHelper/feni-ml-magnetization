import dataclasses
import os
import utils
from dataclasses import field
from pathlib import Path
from typing import Callable, Any, Optional
from opt import GPUOpt, MPIOpt, OMPOpt


@dataclasses.dataclass
class SimulationTask:
    local_input_file: Path = field(default_factory=str)
    gpu: GPUOpt = field(default_factory=GPUOpt)
    mpi: MPIOpt = field(default_factory=MPIOpt)
    omp: OMPOpt = field(default_factory=OMPOpt)
    local_cwd: Path = field(default_factory=str)
    is_test_run: bool = field(default_factory=lambda: False)
    callbacks: list[Callable[[str], None]] = field(default_factory=list)
    ok: bool = field(default_factory=lambda: True)
    nanoparticle: Optional['Nanoparticle'] = field(default_factory=lambda: None)

    def add_callback(self, callback: Callable[[str], None]):
        self.callbacks.append(callback)

    def get_n_threads(self) -> int:
        total_threads: int = 1
        if self.mpi.use:
            total_threads *= self.mpi.n_threads
        if self.omp.use:
            total_threads *= self.omp.n_threads
        return total_threads


class SimulationWrapper:
    """
    Wrapper for MPI LAMMPS
    """

    @staticmethod
    def get_task(
            input_file: Path,
            cwd: Path,
            gpu: GPUOpt = None,
            mpi: MPIOpt = None,
            omp: OMPOpt = None,
    ) -> SimulationTask:
        if gpu is None: gpu = GPUOpt()
        if mpi is None: mpi = MPIOpt()
        if omp is None: omp = OMPOpt()
        return SimulationTask(input_file, gpu, mpi, omp, cwd)

    @staticmethod
    def generate(code: str, file_to_use: Path, sim_params: dict[str, Any] = None) -> SimulationTask:
        """
        Generate the local folder structure and return a simulation task
        :param code:
        :param sim_params:
        :param file_to_use:
        :return:
        """
        assert "{{" not in code, "Not all templates were replaced"
        os.makedirs(file_to_use.parent, exist_ok=True)
        utils.write_local_file(file_to_use, code)
        return SimulationWrapper.get_task(input_file=file_to_use, **sim_params)
