import logging
import os
import random
import re
import subprocess
import time
import typing
from pathlib import Path, PurePath, PurePosixPath
from typing import cast

import numpy as np

import config
import utils
from config import LAMMPS_EXECUTABLE, LOCAL_EXECUTION_PATH, BATCH_INFO_PATH
from execution_queue import ExecutionQueue, SingleExecutionQueue
from remote.slurm_machine import SLURMMachine
from simulation_task import SimulationTask
from template import TemplateUtils
from utils import write_local_file


class SlurmExecutionQueue(SingleExecutionQueue):
    remote: SLURMMachine

    def __init__(self, remote: SLURMMachine):
        super().__init__()
        self.remote = remote

    def submit_toko_script(self, toko_nano_in: PurePosixPath, toko_sim_folder: PurePosixPath, local_sim_folder: Path):
        local_slurm_sh: Path = local_sim_folder / config.SLURM_SH
        toko_slurm_sh: PurePosixPath = toko_sim_folder / config.SLURM_SH
        slurm_code = TemplateUtils.replace_templates(
            TemplateUtils.get_slurm_template(), {
                "lammps_exec": self.remote.lammps_executable.as_posix(),
                "tasks": "1",
                "time": "00:45:00",
                "lammps_input": toko_nano_in.as_posix(),
                "lammps_output": (toko_sim_folder / "log.lammps.bak").as_posix(),
                "cwd": toko_sim_folder.as_posix(),
                "partition": self.remote.partition_to_use,
                "file_tag": toko_slurm_sh.as_posix()
            }
        )
        assert "{{" not in slurm_code, f"Not all templates were replaced in {slurm_code}"
        write_local_file(local_slurm_sh, slurm_code)
        logging.info(f"Copying {config.SLURM_SH} to toko...")
        self.remote.cp_to(local_slurm_sh, toko_slurm_sh)
        logging.info("Queueing job in toko...")
        return self.remote.run_cmd(lambda user, toko_url: ["ssh", f"{user}@{toko_url}",
                                                           f"sh -c 'cd {toko_sim_folder}; {self.remote.sbatch_path} {config.SLURM_SH}'"])

    def simulate_in_toko(self, simulation_task: SimulationTask) -> str:
        input_path = Path(simulation_task.local_input_file)
        local_sim_folder: Path = input_path.parent.resolve()
        toko_sim_folder: PurePosixPath = utils.set_type(PurePosixPath, self.remote.execution_path) / input_path.parent.name
        toko_nano_in: PurePosixPath = toko_sim_folder / input_path.name
        logging.info("Creating simulation folder in toko...")
        logging.debug(f"{toko_sim_folder=}")
        logging.debug(f"{local_sim_folder=}")
        logging.debug(f"{toko_nano_in=}")
        logging.debug("Copying input file...")
        self.remote.cp_to(local_sim_folder, toko_sim_folder, is_folder=True)
        self.remote.copy_alloy_files(local_sim_folder, toko_sim_folder)
        sbatch_output = self.submit_toko_script(toko_nano_in, toko_sim_folder, local_sim_folder)
        jobid = re.match(r"Submitted batch job (\d+)", sbatch_output.decode('utf-8')).group(1)
        self.remote.wait_for_slurm_execution(jobid)
        logging.info("Copying output files from toko to local machine...")
        self.remote.cp_from(toko_sim_folder.as_posix(), local_sim_folder.as_posix(), is_folder=True)
        local_lammps_log = os.path.join(local_sim_folder, "log.lammps")
        with open(local_lammps_log, "r") as f:
            lammps_log = f.read()
        return lammps_log

    def _simulate(self, simulation_task: SimulationTask) -> tuple[SimulationTask, str]:
        try:
            logging.info(
                f"[bold green]TokoExecutionQueue[/bold green] Running [bold yellow]{simulation_task.local_input_file}[/bold yellow] in [cyan]{simulation_task.local_cwd}[/cyan]",
                extra={"markup": True, "highlighter": None})
            result = self.simulate_in_toko(simulation_task)
            return simulation_task, result
        except subprocess.CalledProcessError as e:
            self.print_error(e)
            raise e
        except OSError as e:
            self.print_error(e)
            raise ValueError(f"Is LAMMPS ({LAMMPS_EXECUTABLE}) installed?") from e


def estimate_time(count: int, tasks: int = 1):
    """
    ceil(Count / tasks) * 45 min in d-hh:mm
    From SLURM docs:
    > Acceptable time formats include
    > - "minutes"
    > - "minutes:seconds"
    > - "hours:minutes:seconds"
    > - "days-hours"
    > - "days-hours:minutes"
    > - "days-hours:minutes:seconds"
    :param count: Simulation count
    :param tasks: Thread count
    :return:
    """
    minutes = int(np.ceil(count / tasks) * 45)
    hours = minutes // 60
    days = hours // 24
    remaining_minutes = minutes % 60
    remaining_hours = hours % 24
    return f"{days}-{remaining_hours}:{remaining_minutes}"


class SlurmBatchedExecutionQueue(ExecutionQueue):
    queue: list[SimulationTask]
    completed: list[SimulationTask]
    remote: SLURMMachine

    def enqueue(self, simulation_task: SimulationTask):
        assert isinstance(simulation_task, SimulationTask)
        assert simulation_task.local_input_file is not None
        assert simulation_task.local_cwd is not None
        assert simulation_task.mpi is not None
        assert simulation_task.omp is not None
        assert simulation_task.gpu is not None
        assert simulation_task not in self.queue
        self.queue.append(simulation_task)

    def run(self) -> list[SimulationTask]:
        try:
            self._simulate()
        except Exception as e:
            logging.error(f"Error in {type(self)}: {e}", stack_info=True, exc_info=e)
        return self.completed

    def __init__(self, remote: SLURMMachine, batch_size: int = 10):
        super().__init__()
        self.batch_size = batch_size
        self.remote = remote
        self.queue = []
        self.completed = []

    def submit_toko_script(
            self,
            local_batch_path: Path,
            toko_batch_path: PurePosixPath,
            simulation_count: int,
            n_threads: int = 1,
    ):
        # Create a slurm script to run the commands in parallel
        local_slurm_sh: Path = local_batch_path / config.SLURM_SH
        toko_slurm_sh: PurePosixPath = toko_batch_path / config.SLURM_SH

        slurm_code = TemplateUtils.replace_templates(
            TemplateUtils.get_slurm_multi_template(), {
                "tasks": str(n_threads),
                "time": estimate_time(simulation_count, n_threads),
                "cmd_args": BATCH_INFO_PATH.as_posix(),
                "cwd": toko_batch_path.as_posix(),
                "partition": self.remote.partition_to_use,
                "output": (toko_batch_path / "batch_run.out").as_posix(),
                "file_tag": toko_slurm_sh.as_posix()
            }
        )
        assert "{{" not in slurm_code, f"Not all templates were replaced in {slurm_code}"
        write_local_file(local_slurm_sh, slurm_code)
        logging.info(f"Copying {config.SLURM_SH} to toko...")
        self.remote.cp_to(local_slurm_sh, toko_slurm_sh)
        logging.info("Queueing job in toko...")
        return self.remote.run_cmd(lambda user, toko_url: [
            "ssh",
            f"{user}@{toko_url}",
            f"sh -c 'cd {toko_batch_path}; {self.remote.sbatch_path} {config.SLURM_SH}'"
        ])

    def get_tasks(self, simulation_info):
        # Create a list of commands to run in parallel
        for toko_nano_in, toko_sim_folder, local_sim_folder in simulation_info:
            lammps_log = toko_sim_folder / "log.lammps.bak"
            yield f"sh -c 'cd {toko_sim_folder}; {self.remote.lammps_executable} -in {toko_nano_in} > {lammps_log}'"

    def _simulate(self):
        simulations = self.queue
        local_batch_path, toko_batch_path, simulation_info = self.prepare_scripts(simulations)
        sbatch_output = self.submit_toko_script(
            local_batch_path,
            toko_batch_path,
            simulation_count=len(simulation_info),
            n_threads=self.batch_size,
        )
        self.process_output(local_batch_path, sbatch_output, simulations, toko_batch_path)

    def prepare_scripts(self, simulations: list[SimulationTask]):
        """
        Create a folder in toko with the simulations and a multi.py file to run them in parallel
        Then, copy the folder to toko
        :param simulations: List of simulations to run
        :return:
        """
        simulation_info: list[tuple[PurePosixPath, PurePosixPath, Path]] = []
        # Make new dir in remote execution path "batch_<timestamp>_<random[0-1000]>"
        batch_name: PurePath = PurePath(f"batch_{int(time.time())}_{random.randint(0, 1000)}")
        local_batch_path: Path = LOCAL_EXECUTION_PATH / batch_name
        toko_batch_path: PurePosixPath = cast(PurePosixPath, self.remote.execution_path / batch_name)
        batch_info_path: Path = local_batch_path / BATCH_INFO_PATH
        local_multi_py: Path = config.LOCAL_MULTI_PY
        toko_multi_py: PurePosixPath = toko_batch_path / "multi.py"
        os.mkdir(local_batch_path)

        for i, simulation in enumerate(simulations):
            local_sim_folder: Path = Path(simulation.local_input_file).parent.resolve()
            toko_sim_folder: PurePosixPath = utils.set_type(PurePosixPath, self.remote.execution_path) / local_sim_folder.name
            toko_nano_in: PurePosixPath = toko_sim_folder / simulation.local_input_file.name
            logging.debug(f"Creating simulation folder in toko ({i + 1}/{len(simulations)})...")
            if i == 0:
                logging.debug(f"{toko_sim_folder=}")
                logging.debug(f"{local_sim_folder=}")
                logging.debug(f"{toko_nano_in=}")
                self.remote.copy_alloy_files(local_sim_folder, toko_sim_folder)
            simulation_info.append((toko_nano_in, toko_sim_folder, local_sim_folder))
        tasks = self.get_tasks(simulation_info)
        write_local_file(batch_info_path, "\n".join([
            f"{i + 1}: {os.path.basename(Path(simulation.local_input_file).parent.resolve())} # {shell}"
            for i, (simulation, shell) in enumerate(zip(simulations, tasks))
        ]) + "\n")

        self.remote.cp_multi_to([folder for (_, _, folder) in simulation_info], cast(PurePosixPath, self.remote.execution_path))

        self.remote.cp_to(local_batch_path, toko_batch_path, is_folder=True)
        self.remote.cp_to(local_multi_py, toko_multi_py)
        return local_batch_path, toko_batch_path, simulation_info

    def process_output(self, local_batch_path, sbatch_output, simulations, toko_batch_path):
        jobid = re.match(r"Submitted batch job (\d+)", sbatch_output.decode('utf-8')).group(1)
        self.remote.wait_for_slurm_execution(jobid)
        logging.info("Copying output files from toko to local machine...")
        self.remote.cp_from(toko_batch_path, local_batch_path, is_folder=True)
        files_to_copy_back = []
        callback_info = []
        for i, simulation in enumerate(simulations):
            # Copy each simulation output to the local machine
            local_sim_folder: Path = Path(simulation.local_input_file).parent.resolve()
            toko_sim_folder: Path = Path(os.path.join(self.remote.execution_path, local_sim_folder.name))
            # TokoUtils.copy_file_from_toko(toko_sim_folder.as_posix(), local_sim_folder.as_posix(), is_folder=True)
            local_lammps_log: Path = local_sim_folder / "log.lammps"
            files_to_copy_back.append(toko_sim_folder.as_posix())
            callback_info.append((simulation, local_lammps_log))
        self.remote.cp_multi_from([folder for folder in files_to_copy_back], LOCAL_EXECUTION_PATH)
        for simulation, lammps_log in callback_info:
            self.run_callback(simulation, utils.read_local_file(lammps_log))
            self.completed.append(simulation)
