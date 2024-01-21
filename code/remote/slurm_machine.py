import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import PurePath, PurePosixPath
from typing import Generator, cast

import config
from config import TOKO_PARTITION_TO_USE, TOKO_SBATCH, TOKO_SQUEUE, TOKO_SCONTROL, TOKO_SINFO
from lammpsrun import LammpsRun
from model.live_execution import LiveExecution
from remote.ssh_machine import SSHMachine


@dataclass
class SLURMMachine(SSHMachine):
    """
    A remote machine with a number of cores that we can SSH into and use SLURM
    """

    node_id: int
    partition_to_use: str

    sbatch_path: PurePath
    squeue_path: PurePath
    scontrol_path: PurePath
    sinfo_path: PurePath

    def __str__(self) -> str:
        return f"Machine: {self.name}/{self.partition_to_use}{self.node_id} ({self.cores} cores) with {sum([len(queue) for queue in self.task_queue])} tasks"

    def __init__(
            self,
            name: str,
            cores: int,
            user: str,
            remote_url: str,
            port: int = 22,
            copy_script: PurePath = PurePath('rsync'),
            lammps_executable: PurePosixPath = PurePosixPath('lmp'),
            execution_path: PurePosixPath = PurePosixPath(''),
            partition_to_use: str = TOKO_PARTITION_TO_USE,
            node_id: int = 1,
            sbatch_path: PurePath = TOKO_SBATCH,
            squeue_path: PurePath = TOKO_SQUEUE,
            scontrol_path: PurePath = TOKO_SCONTROL,
            sinfo_path: PurePath = TOKO_SINFO,
    ):
        super().__init__(name, cores, user, remote_url, port, copy_script, lammps_executable, execution_path)
        self.scontrol_path = scontrol_path
        self.squeue_path = squeue_path
        self.sbatch_path = sbatch_path
        self.sinfo_path = sinfo_path
        self.node_id = node_id
        self.partition_to_use = partition_to_use

    def wait_for_slurm_execution(self, jobid):
        logging.info("Waiting for execution to finish in remote...")
        self.run_cmd(lambda user, remote_url: [
            "ssh",
            "-o",
            "ServerAliveInterval=10",
            f"{user}@{remote_url}",
            f"sh -c 'while [ \"$({self.squeue_path} -hj {jobid})\" != \"\" ]; do sleep 1; done'"
        ])

    def get_slurm_jobids(self, of_user: str) -> list[int]:
        """
        Get the job ids of the current user
        Runs: `squeue -u <USER> -h -o %i`
        :return:
        """
        return [int(jobid) for jobid in self.run_cmd(
            lambda user, remote_url: [
                "ssh",
                f"{user}@{remote_url}",
                f"sh -c '{self.squeue_path} -u {of_user} -h -o %i'"
            ]).decode("utf-8").split("\n") if jobid]

    def get_slurm_executing_job_submit_code(self, job_id: int) -> str:
        """
        Get the batch script that was used to run the job
        Runs: `scontrol write batch_script <JOB_ID> -`
        :param job_id:
        :return:
        """
        return self.run_cmd(
            lambda user, remote_url: [
                "ssh",
                f"{user}@{remote_url}",
                f"sh -c '{self.scontrol_path} write batch_script {job_id} -'"
            ]
        ).decode("utf-8")

    def get_file_tag(self, batch_script: str) -> PurePath | None:
        """
        Get the file tag from a batch script
        :param batch_script:
        :return:
        """
        result = re.search(r"## TAG: (.*)", batch_script)
        return None if result is None else PurePath(result.group(1))

    def get_running_tasks(self) -> Generator[LiveExecution, None, None]:
        # List job ids with (in toko with ssh): # squeue -hu fwilliamson -o "%i"
        # Then Get slurm.sh contents with # scontrol write batch_script <JOB_ID> -
        # Then parse {{file_tag}} which is in the last line as a comment with "## TAG: <file_tag>"
        # Then get batch_info.txt from that path
        # Read it
        # Read each simulation and get the current step
        # Return the current step
        job_ids: list[int] = self.get_slurm_jobids(self.user)
        logging.info(f"Found {len(job_ids)} active jobs ({job_ids})")
        for job_id in job_ids:
            logging.info(f"Getting info for job {job_id}")
            batch_script: str = self.get_slurm_executing_job_submit_code(job_id)
            logging.debug(f"Batch script: {batch_script}")
            file_tag: PurePath | None = self.get_file_tag(batch_script)
            logging.debug(f"File tag: {file_tag}")
            if file_tag is None:
                logging.debug("File tag not found")
                continue
            batch_info: PurePath = file_tag.parent / config.BATCH_INFO_PATH
            logging.debug(f"Batch info: {batch_info}")
            try:
                batch_info_content: str = self.read_file(cast(PurePosixPath, batch_info))
            except FileNotFoundError:
                batch_info_content: str = "1: " + (file_tag.parent / config.NANOPARTICLE_IN).as_posix()
            file_read_output, files_to_read = self._read_required_files(batch_info_content)
            total_steps: int = 0
            count: int = 0
            for folder, step, title in self._get_execution_data(file_read_output, files_to_read):
                total_steps += step
                count += 1
                if title != config.FINISHED_JOB:
                    yield folder, step, title
            if batch_info_content.count("\n") > 1:
                logging.debug(f"Found Batch execution")
                yield PurePath(file_tag).parent.as_posix(), total_steps, config.BATCH_EXECUTION + f" ({count})"

    def _get_execution_data(self, file_read_output: dict[str, str], files_to_read: list[tuple[str, str, str]]):
        for folder_name, lammps_log, remote_nano_in in files_to_read:
            try:
                lammps_log_content = file_read_output[lammps_log]
                if "Total wall time" in lammps_log_content:
                    yield folder_name, config.FULL_RUN_DURATION, config.FINISHED_JOB
                    continue
                current_step = LammpsRun.compute_current_step(lammps_log_content)
                logging.debug(f"Current step: {current_step} {folder_name}")
            except subprocess.CalledProcessError:
                current_step = -1
                logging.debug(f"Error getting current step for {folder_name}")
            code = file_read_output[remote_nano_in]
            title = code.split("\n")[0][1:].strip()
            yield folder_name, current_step, title

    def _process_files_to_read(self, batch_info_content) -> Generator[tuple[str, str, str], None, None]:
        for line in batch_info_content.split("\n"):
            if line.strip() == "":
                continue
            index, nano_in = line.split(": ")
            nano_in = PurePath(nano_in.split(" # ")[0] + "/")
            logging.debug(f"Found nano_in: {nano_in}")
            f_name: str = nano_in.parent.name
            folder_name: PurePath = config.TOKO_EXECUTION_PATH / f_name
            lammps_log: PurePath = folder_name / "log.lammps.bak"
            remote_nano_in: PurePath = folder_name / config.NANOPARTICLE_IN
            yield folder_name, lammps_log, remote_nano_in

    def _read_required_files(self, batch_info_content):
        files_to_read: list[tuple[str, str, str]] = list(
            self._process_files_to_read(batch_info_content)
        )
        file_read_output: dict[str, str] = {}
        file_names_to_read = [*[x[1] for x in files_to_read], *[x[2] for x in files_to_read]]
        file_contents = self.read_multiple_files(file_names_to_read)
        logging.info(f"Read {len(file_contents)} files")
        for i, content in enumerate(file_contents):
            file_read_output[file_names_to_read[i]] = content
        return file_read_output, files_to_read
