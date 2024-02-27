import logging
import os
import re
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar, Callable, Generic

import utils
from lammps.lammpsdump import LammpsDump
from lammps.simulation_task import SimulationTask, SimulationWrapper
from utils import generate_random_filename

T = TypeVar('T')
V = TypeVar('V')


class LazyList(Generic[T]):
    def __init__(self, compute_func: Callable[[int], T], length: int) -> None:
        self._compute_func = compute_func
        self.length = length
        self._cache = {}

    def __getitem__(self, index: int) -> T:
        if index < 0:
            raise IndexError("Index out of range")

        if index not in self._cache:
            self._cache[index] = self._compute_func(index)
        return self._cache[index]

    def __len__(self) -> int:
        return self.length


class LazyDict(Generic[T, V]):
    def __init__(self, compute_func: Callable[[V], T], keys: list[V]) -> None:
        self._compute_func = compute_func
        self.v_keys = keys
        self._cache = {}

    def __getitem__(self, index: V) -> T:
        if index not in self.v_keys:
            raise IndexError(f"Index out of range {self.v_keys}")

        if index not in self._cache:
            self._cache[index] = self._compute_func(index)
        return self._cache[index]

    def __len__(self) -> int:
        return len(self.v_keys)

    def __contains__(self, item):
        return item in self.v_keys

    def keys(self):
        return self.v_keys

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
        # "iron.123123.dump"
        step_id_re = re.compile(r"iron.(\d+).dump")
        self.dump_ids: list[int] = [int(step_id_re.match(dump.name).group(1)) for dump in self.expect_dumps]

    @cached_property
    def dumps(self) -> LazyDict:
        return LazyDict(self._parse_dump, self.dump_ids)

    @cached_property
    def lammps_log_filename(self) -> Path:
        return self.cwd / "log.lammps"

    @cached_property
    def bak_lammps_log_filename(self) -> Path:
        return self.cwd / "log.lammps.bak"

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
            pass
        return step

    def get_current_step(self) -> int:
        """
        Get the current step of a lammps log file
        """
        bak_file: Path = self.bak_lammps_log_filename
        if bak_file.exists():
            filename: Path = bak_file
        else:
            filename: Path = self.lammps_log_filename
        try:
            contents: str = utils.read_local_file(filename)
            return LammpsRun.compute_current_step(contents)
        except FileNotFoundError:
            pass
        return -1

    def get_simulation_task(self, test_run: bool) -> SimulationTask:
        """
        Get a simulation task for this run
        """
        sim_task = SimulationWrapper.generate(self.code, file_to_use=self.file_name, sim_params=self.sim_params, test_run=test_run)
        sim_task.add_callback(self.on_post_execution)
        return sim_task

    def on_post_execution(self, result: str | None) -> None:
        if result is None:
            logging.debug(f"Execution failed for lammps run {self.file_name}")
            return
        self.output = result
        logging.debug(f"Finished execution of {self.file_name}")
        logging.debug(f"Lammps run {self} done with {len(self.dumps)} dumps")

    def _parse_dump(self, index: int):
        return LammpsDump(self.expect_dumps[index])

    @staticmethod
    def from_path(path: Path):
        files: list[str] = os.listdir(path)
        nano_in: Path = [path / file for file in files if file.endswith(".in")][0]
        code: str = utils.read_local_file(nano_in)
        dumps: list[Path] = [path / file for file in files if file.endswith(".dump")]
        lr = LammpsRun(code, {'cwd': path}, dumps, nano_in)
        return lr
