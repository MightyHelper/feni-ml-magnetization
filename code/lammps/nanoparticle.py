import logging
import os
import random
import re
import subprocess
import time
from functools import cached_property, cache
from pathlib import Path
from typing import AsyncGenerator

import pandas as pd
from pandas.errors import EmptyDataError

import opt
import template
import utils
from config import config
from config.config import LOCAL_EXECUTION_PATH, FULL_RUN_DURATION, LAMMPS_DUMP_INTERVAL, FE_ATOM, NI_ATOM, \
    NANOPARTICLE_IN
from lammps import feni_ovito, lammpsrun as lr, shapes
from lammps.lammpsdump import LammpsLog
from lammps.simulation_task import SimulationTask
from model.live_execution import LiveExecution
from remote.execution_queue.execution_queue import ExecutionQueue
from remote.machine.machine import Machine
from utils import drop_index


class Nanoparticle:
    """Represents a local nanoparticle"""
    regions: list[shapes.Shape]
    atom_manipulation: list[str]
    run: lr.LammpsRun | None
    region_name_map: dict[str, int] = {}
    magnetism: tuple[float | None, float | None]
    id: str
    title: str
    local_path: Path
    extra_replacements: dict
    coord: pd.DataFrame
    coord_fe: pd.DataFrame
    coord_ni: pd.DataFrame
    psd_p: pd.DataFrame
    psd: pd.DataFrame
    pec: pd.DataFrame
    total: float
    fe_s: float
    ni_s: float
    fe_c: float
    ni_c: float

    def __init__(self, extra_replacements: dict = None, id_x: str = None):
        self.regions = []
        self.atom_manipulation = []
        self.extra_replacements = {} if extra_replacements is None else extra_replacements
        self.rid = random.randint(0, 10000)
        self.title = self.extra_replacements.get("title", "Nanoparticle")
        self.id = self._gen_identifier() if id_x is None else id_x
        self.local_path = (LOCAL_EXECUTION_PATH / self.id).resolve()
        self.run = None

    @staticmethod
    def from_executed(path: Path):
        """
        Loads a nanoparticle from an executed simulation
        :param path: Path to the simulation
        :return: A nanoparticle object
        """
        n = Nanoparticle()
        n.local_path = path.resolve()
        if not os.path.isdir(n.local_path):
            raise Exception(f"Path {n.local_path} is not a directory")
        if not os.path.isfile(n.lammps_log_path):
            raise Exception(f"Path {n.local_path} does not contain a log.lammps file")
        n.id = n.local_path.name
        n.regions = []
        n.atom_manipulation = []
        n.run = lr.LammpsRun.from_path(n.local_path)
        n.extra_replacements = n.run.extra_replacements
        n.region_name_map = {}
        n.title = n.run.title
        return n

    @cached_property
    def coord(self):
        return self.read_coordination(feni_ovito.COORD_FILENAME)

    @cached_property
    def coord_fe(self):
        return self.read_coordination(feni_ovito.COORD_FE_FILENAME)

    @cached_property
    def coord_ni(self):
        return self.read_coordination(feni_ovito.COORD_NI_FILENAME)

    @cached_property
    def psd_p(self):
        return self.read_psd_p(feni_ovito.PARTIAL_G_R_FILENAME)

    @cached_property
    def psd(self):
        return self.read_psd(feni_ovito.G_R_FILENAME)

    @cached_property
    def pec(self):
        return self.read_peh(feni_ovito.PEH_FILENAME)

    @cached_property
    def _surface(self):
        return self.read_surface_atoms(feni_ovito.SURFACE_FILENAME)

    @cached_property
    def total(self):
        return self._surface[0] if len(self._surface) > 0 else 0

    @cached_property
    def fe_s(self):
        return self._surface[1] if len(self._surface) > 1 else 0

    @cached_property
    def ni_s(self):
        return self._surface[2] if len(self._surface) > 2 else 0

    @cached_property
    def fe_c(self):
        return self._surface[3] if len(self._surface) > 3 else 0

    @cached_property
    def ni_c(self):
        return self._surface[4] if len(self._surface) > 4 else 0

    def columns_for_dataset(self):
        # col_order = ["psd", "psd11", "psd12", "psd22", "coordc", "coordc_fe", "coordc_ni", "pec", "fe_s", "ni_s", "fe_c", "ni_c", "n_fe", "n_ni", "tmg"]
        out = [
            drop_index(pd.DataFrame([self.get_descriptor_name()], columns=["name"])),
        ]
        for k, (v, count) in {
            'psd': (self.psd[['psd']].copy(), 100),
            'psd11': (self.psd_p[['1-1']].copy(), 100),
            'psd12': (self.psd_p[['1-2']].copy(), 100),
            'psd22': (self.psd_p[['2-2']].copy(), 100),
            'coordc': (self.coord[['count']].copy(), 100),
            'coordc_fe': (self.coord_fe[['count']].copy(), 100),
            'coordc_ni': (self.coord_ni[['count']].copy(), 100),
            'pec': (self.pec[['count']].copy(), 100),
        }.items():
            out += [self._get_pivoted_df(v, k, expected_row_count=count)]
        for k, v in {
            'fe_s': drop_index(pd.DataFrame([self.fe_s], columns=["fe_s"])),
            'ni_s': drop_index(pd.DataFrame([self.ni_s], columns=["ni_s"])),
            'fe_c': drop_index(pd.DataFrame([self.fe_c], columns=["fe_c"])),
            'ni_c': drop_index(pd.DataFrame([self.ni_c], columns=["ni_c"])),
            'n_fe': drop_index(pd.DataFrame([self.count_atoms_of_type(FE_ATOM)], columns=["n_fe"])) / self.total,
            'n_ni': drop_index(pd.DataFrame([self.count_atoms_of_type(NI_ATOM)], columns=["n_ni"])) / self.total,
            'tmg': drop_index(pd.DataFrame([self.magnetism[0]], columns=["tmg"])),
            'tmg_std': drop_index(pd.DataFrame([self.magnetism[1]], columns=["tmg_std"])),
        }.items():
            out.append(v)
        return pd.concat(out, axis=1)

    @property
    def run_magnetism(self) -> tuple[float | None, float | None]:
        if self.lammps_log.step_count == 0:
            return None, None
        try:
            return self.lammps_log.magnetism['mean'], self.lammps_log.magnetism['std']
        except FileNotFoundError:
            return None, None

    @property
    def run_total_energy(self) -> tuple[float | None, float | None]:
        if self.lammps_log.step_count == 0:
            return None, None
        try:
            return self.lammps_log.total_energy['mean'], self.lammps_log.total_energy['std']
        except FileNotFoundError:
            return None, None

    @property
    def run_magnetic_energy(self) -> tuple[float | None, float | None]:
        if self.lammps_log.step_count == 0:
            return None, None
        try:
            return self.lammps_log.magnetic_energy['mean'], self.lammps_log.magnetic_energy['std']
        except FileNotFoundError:
            return None, None

    @staticmethod
    def _get_pivoted_df(df, name, expected_row_count=100):
        row_count = df.shape[0]
        if row_count > expected_row_count:
            logging.debug("Expanding " + name)
            # noinspection PyTypeChecker
            df["index"] = df.index // (row_count / expected_row_count)
            df = df.groupby("index").mean()
            df.reset_index(inplace=True)
        if row_count < expected_row_count:
            logging.debug("Filling " + name)
            # Expand the range filling the gaps with 0
            new_df = df.reindex(range(expected_row_count), fill_value=0)
            for i in range(row_count):
                new_df.iloc[i] = 0
                new_df.iloc[int((float(expected_row_count) / float(row_count)) * i)] = df.iloc[i]
            df = new_df
        df = df.transpose()
        df.columns = [f"{name}_{int(i) + 1}" for i in df.columns]
        df.index = ["" for _ in df.index]
        return df

    @cache
    def read_coordination(self, filename: Path | str):
        try:
            df = pd.read_csv(self.local_path / filename, delimiter=" ", skiprows=2, header=None)
            df = df.iloc[:, 0:2]
            df.columns = ["coordination", "count"]
            df = df.astype({"coordination": float, "count": float})
            df["coordination"] -= 0.5
        except (FileNotFoundError, EmptyDataError):
            df = pd.DataFrame(columns=["coordination", "count"])
        return df

    @cache
    def read_psd_p(self, filename: Path | str):
        try:
            df = pd.read_csv(self.local_path / filename, delimiter=" ", skiprows=2, header=None)
            df.columns = ["radius", "1-1", "1-2", "2-2"]
            df = df.astype({"radius": float, "1-1": float, "1-2": float, "2-2": float})
        except (FileNotFoundError, EmptyDataError):
            df = pd.DataFrame(columns=["radius", "1-1", "1-2", "2-2"])
        return df

    @cache
    def read_psd(self, filename: Path | str):
        try:
            df = pd.read_csv(self.local_path / filename, delimiter=" ", skiprows=2, header=None)
            df.columns = ["radius", "psd"]
            df = df.astype({"radius": float, "psd": float})
            if len(df) > 100:
                df["index"] = df.index // (len(df) / 100)
                df = df.groupby("index").mean()
                df.reset_index(inplace=True)
                df.drop(columns=["index"], inplace=True)
        except (FileNotFoundError, EmptyDataError):
            df = pd.DataFrame(columns=["radius", "psd"])
        return df

    @cache
    def read_surface_atoms(self, filename: Path | str):
        try:
            with open(self.local_path / filename, "r") as f:
                line = f.readlines()[1].strip()
                return [float(x) for x in re.split(" +", line)]
        except (FileNotFoundError, EmptyDataError):
            df = pd.DataFrame(columns=["type", "count"])
        return df

    @cache
    def read_peh(self, filename: Path | str):
        try:
            df = pd.read_csv(self.local_path / filename, delimiter=" ", skiprows=2, header=None)
            df = df.iloc[:, 0:2]
            df.columns = ["pe", "count"]
            df = df.astype({"pe": float, "count": float})
        except (FileNotFoundError, EmptyDataError):
            df = pd.DataFrame(columns=["pe", "count"])
        return df

    def get_simulation_task(self, test_run: bool = True, **kwargs) -> SimulationTask:
        """
        Generates a simulation task for this nanoparticle
        :param test_run: If true, only one dump will be generated
        :param kwargs: Extra arguments to pass to the lammps run
        :return: A simulation task
        """
        code = self._build_lammps_code(test_run)
        dumps, self.run = self._build_lammps_run(code, kwargs, test_run)
        sim_task = self.run.get_simulation_task(test_run)
        sim_task.add_callback(self.on_post_execution)
        sim_task.nanoparticle = self
        return sim_task

    def schedule_execution(self, execution_queue: ExecutionQueue, test_run: bool = True, **kwargs) -> None:
        """
        Schedules the execution of this nanoparticle
        :param execution_queue: The execution queue to use
        :param test_run: If true, only one dump will be generated
        :param kwargs: Extra arguments to pass to the lammps run
        """
        execution_queue.enqueue(self.get_simulation_task(test_run, **kwargs))

    def on_post_execution(self, result: str | None) -> None:
        """
        Callback for when the execution is finished
        :return:
        """
        if result is None:
            lammps_log_path: Path = self.lammps_log_path
            logging.info(f"{self.local_path=} {lammps_log_path}")
            lammps_log: str = utils.read_local_file(lammps_log_path)
            logging.warning(f"Run for nanoparticle {self.title} failed. LAMMPS Log:\n{lammps_log}")
            return
        if FULL_RUN_DURATION in self.run.dumps:
            self.lammps_log.save_mag_to_file(self.local_path / "magnetism.txt", digits=4)
            feni_ovito.FeNiOvitoParser.parse(
                filenames={
                    'base_path': self.local_path,
                    'dump': os.path.basename(self.run.dumps[FULL_RUN_DURATION].path)
                }
            )

    @cached_property
    def lammps_log_path(self) -> Path:
        return self.local_path / "log.lammps"

    @cached_property
    def lammps_log(self) -> LammpsLog:
        return LammpsLog(self.lammps_log_path)

    def _build_lammps_run(self, code, kwargs, test_run):
        lammps_run = lr.LammpsRun(
            code, {
                "omp": opt.OMPOpt(use=False, n_threads=2),
                "mpi": opt.MPIOpt(use=False, hw_threads=False, n_threads=4),
                "cwd": self.local_path,
                **kwargs
            },
            dumps := [f"iron.{i}.dump" for i in
                      ([0] if test_run else [*range(0, FULL_RUN_DURATION + 1, LAMMPS_DUMP_INTERVAL)])],
            file_name=self.local_path / NANOPARTICLE_IN
        )
        return dumps, lammps_run

    def _build_lammps_code(self, test_run):
        return template.TemplateUtils.replace_with_doc(template.TemplateUtils.get_lammps_template(), {
            "region": self.get_region(),
            "run_steps": str(0 if test_run else FULL_RUN_DURATION),
            "title": f"{self.title}",
            **self.extra_replacements
        }, 'json_description')

    def _gen_identifier(self):
        return f"simulation_{time.time()}_{self.rid}"

    @cached_property
    def get_simulation_date(self) -> float:
        return float(self.id.split("_")[1])

    @cached_property
    def magnetism(self):
        return self.run_magnetism

    def get_region(self):
        out = ""
        for i in self.atom_manipulation:
            if isinstance(i, str):
                out += i + "\n"
            elif isinstance(i, shapes.Shape):
                out += i.get_region(f"reg{self.regions.index(i)}") + "\n"
                raise Exception("Shapes are supported but not allowed")
            else:
                raise Exception(f"Unknown type: {type(i)}")
        return out

    def plot(self):
        if self.run is None:
            raise Exception("Run not executed")
        self.run.dumps[0].plot()

    def plot_tmg(self):
        self.lammps_log.plot_tmg(self.title)

    def count_atoms_of_type(self, atom_type, dump_idx=0):
        if dump_idx not in self.run.dumps:
            logging.warning(f"Dump {dump_idx} not found (Available dumps: {self.run.dumps.keys()}) - {self.run.dumps=}")
            raise Exception(f"Dump {dump_idx} not found (Available dumps: {self.run.dumps.keys()}) - {self.run.dumps=}")
        return self.run.dumps[dump_idx].count_atoms_of_type(atom_type)

    def total_atoms(self, dump_idx=0):
        return self.run.dumps[dump_idx].dump['number_of_atoms']

    def __str__(self):
        return f"Nanoparticle(regions={len(self.regions)} items, atom_manipulation={len(self.atom_manipulation)} items, title={self.title})"

    def __repr__(self):
        return self.__str__()

    def get_descriptor_name(self):
        result = self.title.split("/")[-1]
        logging.debug(f"Descriptor name: {self.title} => {result}")
        return result

    def get_full_coord(self):
        out = self.coord.copy()
        out["count_fe"] = self.coord_fe["count"]
        out["count_ni"] = self.coord_ni["count"]
        return out

    def asdict(self):
        try:
            if self.is_ok():
                return {
                    "ok": self.is_ok(),
                    "key": self.id,
                    "title": self.title,
                    "np": self,
                    "fe": self.count_atoms_of_type(FE_ATOM),
                    "ni": self.count_atoms_of_type(NI_ATOM),
                    "total": self.total_atoms(),
                    "ratio_fe": self.atom_type_ratio(FE_ATOM),
                    "ratio_ni": self.atom_type_ratio(NI_ATOM),
                    "emag": self.run_magnetic_energy,
                    "teng": self.run_total_energy,
                    "tmg": self.run_magnetism
                }
            else:
                return {
                    "ok": self.is_ok(),
                    "key": self.id,
                    "title": self.title,
                    "np": self,
                    "mag": self.magnetism
                }
        except Exception as e:
            logging.warning(f"Error getting nanoparticle data: {type(e)} {e}", stack_info=True)
            return {
                "ok": False,
                "key": self.id,
                "title": self.title,
                "np": self,
                "mag": float('nan')
            }

    def atom_type_ratio(self, atom_type: int) -> float:
        return self.count_atoms_of_type(atom_type) / self.total_atoms()

    def is_ok(self):
        return len(self.run.dumps) > 0

    def plot_tmg_vs_teng(self):
        self.lammps_log.plot_tmg_vs_teng(self.title)

    def plot_teng_hist(self):
        self.lammps_log.plot_teng_hist(self.title)

    def is_weak(self):
        return self.magnetism[1] / self.magnetism[0] > config.MAX_MAGNETISM_VARIANCE


class RunningExecutionLocator:
    @staticmethod
    def get_running_windows(from_windows: bool = True):
        # wmic.exe process where "name='python.exe'" get commandline, disable stderr

        if from_windows:
            path = "wmic.exe"
        else:
            path = "/mnt/c/Windows/System32/Wbem/wmic.exe"
        result = subprocess.check_output(
            [
                path,
                "process",
                "where",
                f"name='{config.LOCAL_LAMMPS_NAME_WINDOWS}'",
                "get",
                "commandline"
            ],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').replace("\r", "").split("\n")
        logging.debug(f"WMIC Result: {result}")
        result = [x.strip() for x in result if x.strip() != ""]
        xv: str = ""
        for execution in {xv for result in result if
                          "-in" in result and (xv := re.sub(".*?(-in (.*))\n?", "\\2", result).strip()) != ""}:
            logging.debug(f"Found execution: {execution}")
            folder_name = Path(execution).parent
            nano = Nanoparticle.from_executed(folder_name)
            yield folder_name, nano.run.get_current_step(), nano.title

    @staticmethod
    async def get_running_executions(machine: Machine) -> AsyncGenerator[LiveExecution, None]:
        async for item in await machine.get_running_tasks():  # TODO: Fixme if live exec not working
            yield item

    @staticmethod
    def get_nth_path_element(path: str, n: int) -> str:
        return path.split("/")[n]

    @staticmethod
    def get_upto_nth_path_element(path: str, n: int) -> str:
        return "/".join(path.split("/")[:n])
