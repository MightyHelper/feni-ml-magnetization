import logging
import re
from pathlib import Path, PurePosixPath

import numpy as np
from asyncssh import SSHCompletedProcess

import utils
from config import config
from config.config import BATCH_INFO
from remote.machine.local_machine import LocalMachine
from remote.machine.slurm_machine import SLURMMachine
from remote.machine.ssh_machine import SSHBatchedExecutionQueue
from template import TemplateUtils
from utils import write_local_file

SINGLE_SIMULATION_TIME = 45


def estimate_slurm_time(count: int, tasks: int = 1, machine_power: float = 1.0):
    """
    :param count: Simulation count
    :param tasks: Thread count
    :param machine_power: Machine power multiplier
    :return:
    """
    minutes = estimate_minutes(count, tasks, machine_power)
    return minutes_to_slurm(minutes)


def minutes_to_slurm(minutes: float):
    """
    From SLURM docs:
    > Acceptable time formats include
    > - "minutes"
    > - "minutes:seconds"
    > - "hours:minutes:seconds"
    > - "days-hours"
    > - "days-hours:minutes"
    > - "days-hours:minutes:seconds"
    :param minutes: Minutes to convert
    :return:
    """
    hours = minutes // 60
    days = hours // 24
    remaining_minutes = minutes % 60
    remaining_hours = hours % 24
    return f"{days}-{remaining_hours}:{remaining_minutes}"


def estimate_minutes(count: int, tasks: int, machine_power: float) -> float:
    """
    ceil(Count / tasks) * SINGLE_SIMULATION_TIME min
    :param count: Simulation count
    :param machine_power: Machine power multiplier
    :param tasks: Thread count
    :return:
    """
    return int(np.ceil(count / tasks) * SINGLE_SIMULATION_TIME / machine_power)


class SlurmBatchedExecutionQueue(SSHBatchedExecutionQueue):
    remote: SLURMMachine
    def __init__(self, remote: SLURMMachine, local: LocalMachine, batch_size: int = 10):
        super().__init__(remote, local, batch_size)
        self.batch_size = batch_size
        self.remote = remote
        self.queue = []
        self.completed = []



    def _generate_local_run_file(self, batch_name: str, n_threads: int, simulation_count: int):
        remote_batch_path: PurePosixPath = utils.set_type(PurePosixPath, self.remote.execution_path) / batch_name
        local_run_script_path: Path = self._get_local_exec_child(batch_name) / config.RUN_SH
        script_code: str = TemplateUtils.replace_templates(
            TemplateUtils.get_slurm_multi_template(), {
                "tasks": str(n_threads),
                "time": estimate_slurm_time(simulation_count, n_threads, self.remote.single_core_performance),
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
        self.local.run_cmd(["chmod", "u+x", local_run_script_path])


    async def submit_remote_batch(self, batch_name: str):
        logging.info("Queueing job in toko...")
        sbatch: SSHCompletedProcess = await self.remote.run_cmd(f"sh -c 'cd {self.remote.execution_path / batch_name}; {self.remote.sbatch_path} {config.RUN_SH}'")
        jobid: int = re.match(r"Submitted batch job (\d+)", sbatch.stdout).group(1)
        await self.remote.wait_for_slurm_execution(jobid)
