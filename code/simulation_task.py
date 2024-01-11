import dataclasses
import os
from dataclasses import field
from pathlib import Path
from typing import Callable, Any, Optional

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
            input_file: str = "in.melt",
            gpu: GPUOpt = None,
            mpi: MPIOpt = None,
            omp: OMPOpt = None,
            cwd: str = './lammps_output',
    ) -> SimulationTask:
        if gpu is None: gpu = GPUOpt()
        if mpi is None: mpi = MPIOpt()
        if omp is None: omp = OMPOpt()
        return SimulationTask(input_file, gpu, mpi, omp, cwd)

    @staticmethod
    def generate(code: str, sim_params: dict[str, Any] = None, file_to_use: str = '/tmp/in.melt') -> SimulationTask:
        """
        Generate the local folder structure and return a simulation task
        :param code:
        :param sim_params:
        :param file_to_use:
        :return:
        """
        assert "{{" not in code, "Not all templates were replaced"
        path = Path(file_to_use)
        if not path.parent.exists():
            if not path.parent.parent.exists():
                os.mkdir(path.parent.parent)
            os.mkdir(path.parent)
        with open(file_to_use, 'w') as f:
            f.write(code)
        return SimulationWrapper.get_task(input_file=file_to_use, **sim_params)
