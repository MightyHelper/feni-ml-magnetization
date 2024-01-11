import logging
import os
import platform
import random
import re
import subprocess
import time
from typing import Generator

import pandas as pd

import config
import feni_mag
import feni_ovito
import lammpsrun as lr
import opt
import shapes
import template
import toko_utils
import utils
from config import LOCAL_EXECUTION_PATH, FULL_RUN_DURATION, LAMMPS_DUMP_INTERVAL, FE_ATOM, NI_ATOM, \
    BATCH_EXECUTION, NANOPARTICLE_IN
from execution_queue import ExecutionQueue
from simulation_task import SimulationTask
from toko_utils import toko_path_join
from utils import drop_index, realpath

FINISHED_JOB = "Finished job"


class Nanoparticle:
    """Represents a nanoparticle"""
    regions: list[shapes.Shape]
    atom_manipulation: list[str]
    run: lr.LammpsRun | None
    region_name_map: dict[str, int] = {}
    magnetism: tuple[float | None, float | None]
    id: str
    title: str
    path: str
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
        self.path = realpath(LOCAL_EXECUTION_PATH + "/" + self.id) + "/"
        self.run = None
        self.magnetism = (None, None)

    def hardcoded_g_r_crop(self, g_r):
        return g_r[0:60]

    @staticmethod
    def from_executed(path: str):
        """
        Loads a nanoparticle from an executed simulation
        :param path: Path to the simulation
        :return: A nanoparticle object
        """
        n = Nanoparticle()
        n.path = realpath(path)
        if not os.path.isdir(n.path):
            raise Exception(f"Path {n.path} is not a directory")
        lammps_log = n.get_lammps_log_path()
        if not os.path.isfile(lammps_log):
            raise Exception(f"Path {n.path} does not contain a log.lammps file")
        n.id = n.path.split("/")[-1]
        n.regions = []
        n.atom_manipulation = []
        n.extra_replacements = {
            'in_toko': os.path.isfile(n.path + "/" + config.SLURM_SH)
        }
        n.run = lr.LammpsRun.from_path(n.path)
        n.region_name_map = {}
        n.magnetism = n.get_magnetism()
        n.title = n.run.code.split("\n")[0][1:].strip()
        n.coord = n.read_coordination(feni_ovito.COORD_FILENAME)
        n.coord_fe = n.read_coordination(feni_ovito.COORD_FE_FILENAME)
        n.coord_ni = n.read_coordination(feni_ovito.COORD_NI_FILENAME)
        n.psd_p = n.hardcoded_g_r_crop(n.read_psd_p(feni_ovito.PARTIAL_G_R_FILENAME))
        n.psd = n.hardcoded_g_r_crop(n.read_psd(feni_ovito.G_R_FILENAME))
        n.pec = n.read_peh(feni_ovito.PEH_FILENAME)
        surf = n.read_surface_atoms(feni_ovito.SURFACE_FILENAME)
        n.total = surf[0] if len(surf) > 0 else 0
        n.fe_s = surf[1] if len(surf) > 1 else 0
        n.ni_s = surf[2] if len(surf) > 2 else 0
        n.fe_c = surf[3] if len(surf) > 3 else 0
        n.ni_c = surf[4] if len(surf) > 4 else 0
        return n

    def columns_for_dataset(self):
        # col_order = ["psd", "psd11", "psd12", "psd22", "coordc", "coordc_fe", "coordc_ni", "pec", "fe_s", "ni_s", "fe_c", "ni_c", "n_fe", "n_ni", "tmg"]
        out = [
            drop_index(pd.DataFrame([self.get_descriptor_name()], columns=["name"])),
        ]
        for k, (v, count) in {
            'psd': (self.psd[['psd']].copy(), 60),
            'psd11': (self.psd_p[['1-1']].copy(), 60),
            'psd12': (self.psd_p[['1-2']].copy(), 60),
            'psd22': (self.psd_p[['2-2']].copy(), 60),
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
            out += [v]

        # Concat columns of all dataframes
        return pd.concat(out, axis=1)

    def _get_pivoted_df(self, df, name, expected_row_count=100):
        row_count = df.shape[0]
        if row_count > expected_row_count:
            logging.debug("Expanding " + name)
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

    def read_coordination(self, filename):
        try:
            df = pd.read_csv(self.path + "/" + filename, delimiter=" ", skiprows=2, header=None)
            df = df.iloc[:, 0:2]
            df.columns = ["coordination", "count"]
            df = df.astype({"coordination": float, "count": float})
            df["coordination"] -= 0.5
        except FileNotFoundError:
            df = pd.DataFrame(columns=["coordination", "count"])
        return df

    def read_psd_p(self, filename):
        try:
            df = pd.read_csv(self.path + "/" + filename, delimiter=" ", skiprows=2, header=None)
            df.columns = ["radius", "1-1", "1-2", "2-2"]
            df = df.astype({"radius": float, "1-1": float, "1-2": float, "2-2": float})
        except FileNotFoundError:
            df = pd.DataFrame(columns=["radius", "1-1", "1-2", "2-2"])
        return df

    def read_psd(self, filename):
        try:
            df = pd.read_csv(self.path + "/" + filename, delimiter=" ", skiprows=2, header=None)
            df.columns = ["radius", "psd"]
            df = df.astype({"radius": float, "psd": float})
            if len(df) > 100:
                df["index"] = df.index // (len(df) / 100)
                df = df.groupby("index").mean()
                df.reset_index(inplace=True)
                df.drop(columns=["index"], inplace=True)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["radius", "psd"])
        return df

    def read_surface_atoms(self, filename):
        try:
            with open(self.path + "/" + filename, "r") as f:
                line = f.readlines()[1].strip()
                return [float(x) for x in re.split(" +", line)]
        except FileNotFoundError:
            df = pd.DataFrame(columns=["type", "count"])
        return df

    def read_peh(self, filename):
        try:
            df = pd.read_csv(self.path + "/" + filename, delimiter=" ", skiprows=2, header=None)
            df = df.iloc[:, 0:2]
            df.columns = ["pe", "count"]
            df = df.astype({"pe": float, "count": float})
        except FileNotFoundError:
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
        sim_task = self.run.get_simulation_task()
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
            lammps_log_path: str = self.get_lammps_log_path()
            print(f"{self.path=} {lammps_log_path}")
            lammps_log: str = utils.read_local_file(lammps_log_path)
            logging.warning(f"Run for nanoparticle {self.title} failed. LAMMPS Log:\n{lammps_log}")
            return
        if FULL_RUN_DURATION in self.run.dumps:
            feni_mag.MagnetismExtractor.extract_magnetism(self.get_lammps_log_path(),
                                                          out_mag=self.path + "/magnetism.txt", digits=4)
            feni_ovito.FeNiOvitoParser.parse(
                filenames={'base_path': self.path, 'dump': os.path.basename(self.run.dumps[FULL_RUN_DURATION].path)})
            self.magnetism = self.get_magnetism()

    def get_lammps_log_path(self) -> str:
        return os.path.join(self.path, "log.lammps")

    def _build_lammps_run(self, code, kwargs, test_run):
        lammps_run = lr.LammpsRun(
            code, {
                "omp": opt.OMPOpt(use=False, n_threads=2),
                "mpi": opt.MPIOpt(use=False, hw_threads=False, n_threads=4),
                "cwd": self.path,
                **kwargs
            },
            dumps := [f"iron.{i}.dump" for i in
                      ([0] if test_run else [*range(0, FULL_RUN_DURATION + 1, LAMMPS_DUMP_INTERVAL)])],
            file_name=self.path + NANOPARTICLE_IN
        )
        return dumps, lammps_run

    def _build_lammps_code(self, test_run):
        return template.TemplateUtils.replace_templates(template.TemplateUtils.get_lammps_template(), {
            "region": self.get_region(),
            "run_steps": str(0 if test_run else FULL_RUN_DURATION),
            "title": f"{self.title}",
            **self.extra_replacements
        })

    def _gen_identifier(self):
        return f"simulation_{time.time()}_{self.rid}"

    def get_simulation_date(self):
        return self.id.split("_")[1]

    def get_magnetism(self):
        try:
            with open(self.path + "/magnetism.txt", "r") as f:
                lines = f.readlines()
            result = [float(x) for x in lines[1].split(" ")]
            return result[0], result[1]
        except FileNotFoundError:
            return float('nan'), float('nan')

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

    def count_atoms_of_type(self, atom_type, dump_idx=0):
        if dump_idx not in self.run.dumps:
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
                    "ratio_fe": (self.count_atoms_of_type(FE_ATOM) / self.total_atoms()),
                    "ratio_ni": (self.count_atoms_of_type(NI_ATOM) / self.total_atoms()),
                    "mag": self.magnetism
                }
            else:
                return {
                    "ok": self.is_ok(),
                    "key": self.id,
                    "title": self.title,
                    "np": self,
                    "fe": -1,
                    "ni": -1,
                    "total": -1,
                    "ratio_fe": -1,
                    "ratio_ni": -1,
                    "mag": self.magnetism
                }
        except Exception as e:
            logging.warning(f"Error getting nanoparticle data: {type(e)} {e}", stack_info=True)
            return {
                "ok": False,
                "key": self.id,
                "title": self.title,
                "np": self,
                "fe": -1,
                "ni": -1,
                "total": -1,
                "ratio_fe": -1,
                "ratio_ni": -1,
                "mag": float('nan')
            }

    def is_ok(self):
        return len(self.run.dumps) > 0


class RunningExecutionLocator:
    @staticmethod
    def get_running_windows(from_windows: bool = True):
        # wmic.exe process where "name='python.exe'" get commandline, disable stderr

        if from_windows:
            path = "wmic.exe"
        else:
            path = "/mnt/c/Windows/System32/Wbem/wmic.exe"
        result = subprocess.check_output([path, "process", "where", "name='python.exe'", "get", "commandline"],
                                         stderr=subprocess.DEVNULL).decode('utf-8').split("\n")
        logging.debug(f"WMIC Result: {result}")
        result = [x.strip() for x in result if x.strip() != ""]
        for execution in {x for result in result if
                          "-in" in result and (x := re.sub(".*?(-in (.*))\n?", "\\2", result).strip()) != ""}:
            logging.debug(f"Found execution: {execution}")
            folder_name = RunningExecutionLocator.get_nth_path_element(execution.replace("\\", "/"), -1)
            nano = Nanoparticle.from_executed(folder_name)
            yield folder_name, nano.run.get_current_step(), nano.title

    @staticmethod
    def get_running_executions(in_toko: bool = False) -> Generator[tuple[str, int, str], None, None]:
        if not in_toko:
            if platform.system() == "Windows":
                yield from RunningExecutionLocator.get_running_windows(True)
            elif platform.system() == "Linux":
                try:
                    yield from RunningExecutionLocator.get_running_windows(False)
                except FileNotFoundError:
                    pass
                yield from RunningExecutionLocator.get_running_linux()
            else:
                raise Exception(f"Unknown system: {platform.system()}")
        else:
            yield from RunningExecutionLocator.get_running_toko()

    @staticmethod
    def get_running_linux():
        ps_result = os.popen("ps -ef | grep " + config.LAMMPS_EXECUTABLE).readlines()
        for execution in {x for result in ps_result if (x := re.sub(".*?(-in (.*))?\n", "\\2", result)) != ""}:
            folder_name = RunningExecutionLocator.get_upto_nth_path_element(execution, -1)
            try:
                nano = Nanoparticle.from_executed(folder_name)
                yield folder_name, nano.run.get_current_step(), nano.title
            except Exception:
                pass

    @staticmethod
    def get_nth_path_element(path: str, n: int) -> str:
        return path.split("/")[n]

    @staticmethod
    def get_upto_nth_path_element(path: str, n: int) -> str:
        return "/".join(path.split("/")[:n])

    @staticmethod
    def get_running_toko():
        # List job ids with (in toko with ssh): # squeue -hu fwilliamson -o "%i"
        # Then Get slurm.sh contents with # scontrol write batch_script <JOB_ID> -
        # Then parse {{file_tag}} which is in the last line as a comment with "## TAG: <file_tag>"
        # Then get batch_info.txt from that path
        # Read it
        # Read each simulation and get the current step
        # Return the current step
        job_ids: list[int] = toko_utils.TokoUtils.get_my_jobids()
        logging.info(f"Found {len(job_ids)} active jobs ({job_ids})")
        for job_id in job_ids:
            logging.info(f"Getting info for job {job_id}")
            batch_script: str = toko_utils.TokoUtils.get_batch_script(job_id)
            logging.debug(f"Batch script: {batch_script}")
            file_tag: str | None = toko_utils.TokoUtils.get_file_tag(batch_script)
            logging.debug(f"File tag: {file_tag}")
            if file_tag is None:
                logging.debug("File tag not found")
                continue
            batch_info: str = toko_path_join(os.path.dirname(file_tag), config.TOKO_BATCH_INFO_PATH)
            logging.debug(f"Batch info: {batch_info}")
            try:
                batch_info_content: str = toko_utils.TokoUtils.read_file(batch_info)
            except FileNotFoundError:
                batch_info_content: str = "1: " + toko_path_join(os.path.dirname(file_tag), NANOPARTICLE_IN)
            file_read_output, files_to_read = RunningExecutionLocator._read_required_files(batch_info_content)
            total_steps: int = 0
            count: int = 0
            for folder, step, title in RunningExecutionLocator._get_execution_data(file_read_output, files_to_read):
                total_steps += step
                count += 1
                if title != FINISHED_JOB:
                    yield folder, step, title
            if batch_info_content.count("\n") > 1:
                logging.debug(f"Found Batch execution")
                yield os.path.dirname(file_tag), total_steps, BATCH_EXECUTION + f" ({count})"


    @staticmethod
    def _read_required_files(batch_info_content):
        files_to_read: list[tuple[str, str, str]] = list(
            RunningExecutionLocator._process_files_to_read(batch_info_content)
        )
        file_read_output: dict[str, str] = {}
        file_names_to_read = [*[x[1] for x in files_to_read], *[x[2] for x in files_to_read]]
        file_contents = toko_utils.TokoUtils.read_multiple_files(file_names_to_read)
        logging.info(f"Read {len(file_contents)} files")
        for i, content in enumerate(file_contents):
            file_read_output[file_names_to_read[i]] = content
        return file_read_output, files_to_read

    @staticmethod
    def _get_execution_data(file_read_output: dict[str, str], files_to_read: list[tuple[str, str, str]]):
        for folder_name, lammps_log, remote_nano_in in files_to_read:
            try:
                lammps_log_content = file_read_output[lammps_log]
                if "Total wall time" in lammps_log_content:
                    yield folder_name, config.FULL_RUN_DURATION, FINISHED_JOB
                    continue
                current_step = lr.LammpsRun.compute_current_step(lammps_log_content)
                logging.debug(f"Current step: {current_step} {folder_name}")
            except subprocess.CalledProcessError:
                current_step = -1
                logging.debug(f"Error getting current step for {folder_name}")
            code = file_read_output[remote_nano_in]
            title = code.split("\n")[0][1:].strip()
            yield folder_name, current_step, title

    @staticmethod
    def _process_files_to_read(batch_info_content) -> Generator[tuple[str, str, str], None, None]:
        for line in batch_info_content.split("\n"):
            if line.strip() == "":
                continue
            index, nano_in = line.split(": ")
            nano_in = nano_in.split(" # ")[0] + "/"
            logging.debug(f"Found nano_in: {nano_in}")
            f_name: str = os.path.basename(os.path.dirname(nano_in))
            folder_name: str = toko_path_join(config.TOKO_EXECUTION_PATH, f_name)
            lammps_log: str = toko_path_join(folder_name, "log.lammps")
            remote_nano_in: str = toko_path_join(folder_name, NANOPARTICLE_IN)
            yield folder_name, lammps_log, remote_nano_in
