import logging
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

import asyncssh
import numpy as np
from asyncssh import SSHCompletedProcess

import utils
from config.config import LAMMPS_EXECUTABLE, BATCH_INFO
from config import config
from remote.execution_queue.execution_queue import SingleExecutionQueue
from remote.machine.local_machine import LocalMachine
from remote.machine.slurm_machine import SLURMMachine
from remote.machine.ssh_machine import SSHBatchedExecutionQueue
from lammps.simulation_task import SimulationTask
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


class SlurmBatchedExecutionQueue(SSHBatchedExecutionQueue):
    remote: SLURMMachine
    def __init__(self, remote: SLURMMachine, local: LocalMachine, batch_size: int = 10):
        super().__init__(remote, local, batch_size)
        self.batch_size = batch_size
        self.remote = remote
        self.queue = []
        self.completed = []

    def enqueue(self, simulation_task: SimulationTask):
        assert isinstance(simulation_task, SimulationTask)
        assert simulation_task.local_input_file is not None
        assert simulation_task.local_cwd is not None
        assert simulation_task.mpi is not None
        assert simulation_task.omp is not None
        assert simulation_task.gpu is not None
        assert simulation_task not in self.queue
        self.queue.append(simulation_task)


    def _generate_local_run_file(self, batch_name: str, n_threads: int, simulation_count: int):
        remote_batch_path: PurePosixPath = utils.set_type(PurePosixPath, self.remote.execution_path) / batch_name
        local_run_script_path: Path = self._get_local_exec_child(batch_name) / config.RUN_SH
        script_code: str = TemplateUtils.replace_templates(
            TemplateUtils.get_slurm_multi_template(), {
                "tasks": str(n_threads),
                "time": estimate_time(simulation_count, n_threads),
                "cmd_args": str(BATCH_INFO),
                "cwd": str(remote_batch_path),
                "partition": self.remote.partition_to_use,
                "output": str(remote_batch_path / "batch_run.out"),
                "file_tag": str(remote_batch_path / config.RUN_SH),
            }
        )
        assert "{{" not in script_code, f"Not all templates were replaced in {script_code} for {self}"
        write_local_file(local_run_script_path, script_code)
        # Change permission u+x
        self.local.run_cmd(lambda: ["chmod", "u+x", local_run_script_path])


    async def submit_remote_batch(self, batch_name: str, connection: asyncssh.SSHClientConnection):
        logging.info("Queueing job in toko...")
        sbatch: SSHCompletedProcess = await connection.run(f"sh -c 'cd {self.remote.execution_path / batch_name}; {self.remote.sbatch_path} {config.RUN_SH}'")
        jobid: int = re.match(r"Submitted batch job (\d+)", sbatch.stdout).group(1)
        await self.remote.wait_for_slurm_execution(connection, jobid)
