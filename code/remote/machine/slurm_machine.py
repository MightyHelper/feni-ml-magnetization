import asyncio
import logging
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import PurePath, PurePosixPath, Path
from typing import Generator, cast, AsyncGenerator

import utils
from config import config
from config.config import TOKO_PARTITION_TO_USE, TOKO_SBATCH, TOKO_SQUEUE, TOKO_SCONTROL, TOKO_SINFO
from lammps.lammpsrun import LammpsRun
from model.live_execution import LiveExecution
from remote.machine.ssh_machine import SSHMachine


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
        return f"Machine: {self.name}{self.node_id} ({self.cores} cores)"

    def __init__(
        self,
        name: str,
        cores: int,
        user: str,
        remote_url: str,
        port: int = 22,
        password: str | None = None,
        lammps_executable: PurePosixPath = PurePosixPath('lmp'),
        execution_path: PurePosixPath = PurePosixPath(''),
        partition_to_use: str = TOKO_PARTITION_TO_USE,
        node_id: int = 1,
        sbatch_path: PurePath = TOKO_SBATCH,
        squeue_path: PurePath = TOKO_SQUEUE,
        scontrol_path: PurePath = TOKO_SCONTROL,
        sinfo_path: PurePath = TOKO_SINFO,
        launch_time: float = 0.0,
        single_core_completion_time: float = 1.0
    ):
        super().__init__(
            name=name,
            cores=cores,
            user=user,
            remote_url=remote_url,
            port=port,
            password=password,
            lammps_executable=lammps_executable,
            execution_path=execution_path,
            launch_time=launch_time,
            single_core_completion_time=single_core_completion_time
        )
        self.scontrol_path = scontrol_path
        self.squeue_path = squeue_path
        self.sbatch_path = sbatch_path
        self.sinfo_path = sinfo_path
        self.node_id = node_id
        self.partition_to_use = partition_to_use

    async def wait_for_slurm_execution(self, jobid: int):
        logging.info("Waiting for execution to finish in remote...")
        return await self.run_retryable_cmd(f"sh -c 'while [ \"$({self.squeue_path} -hj {jobid})\" != \"\" ]; do sleep 1; done'")

    async def get_slurm_jobids(self, of_user: str | None) -> list[int]:
        """
        Get the job ids of the current user
        Runs: `squeue -u <USER> -h -o %i`
        :return:
        """
        of_user = f"-u {of_user}" if of_user is not None else ""
        return [int(jobid) for jobid in (await self.run_retryable_cmd(f"sh -c '{self.squeue_path} {of_user} -h -o %i'")).stdout.split("\n") if jobid]

    async def get_slurm_executing_job_submit_code(self, job_id: int) -> str:
        """
        Get the batch script that was used to run the job
        Runs: `scontrol write batch_script <JOB_ID> -`
        :param job_id:
        :return:
        """
        return (await self.run_retryable_cmd(
            f"sh -c '{self.scontrol_path} write batch_script {job_id} -'"
        )).stdout

    def get_file_tag(self, batch_script: str) -> PurePath | None:
        """
        Get the file tag from a batch script
        :param batch_script:
        :return:
        """
        result = re.search(r"## TAG: (.*)", batch_script)
        return None if result is None else PurePath(result.group(1))

    async def get_running_tasks(self) -> AsyncGenerator[LiveExecution, None]:
        # List job ids with (in toko with ssh): # squeue -hu fwilliamson -o "%i"
        # Then Get slurm.sh contents with # scontrol write batch_script <JOB_ID> -
        # Then parse {{file_tag}} which is in the last line as a comment with "## TAG: <file_tag>"
        # Then get batch_info.txt from that path
        # Read it
        # Read each simulation and get the current step
        # Return the current step
        connected: bool = False
        if self.connection is None:
            await self.connect(start_sftp=True)
            connected = True

        job_ids: list[int] = await self.get_slurm_jobids(self.user)
        logging.info(f"Found {len(job_ids)} active jobs ({job_ids})")
        for job_id in job_ids:
            logging.info(f"Getting info for job {job_id}")
            batch_script: str = await self.get_slurm_executing_job_submit_code(job_id)
            logging.debug(f"Batch script: {batch_script}")
            file_tag: PurePath | None = self.get_file_tag(batch_script)
            logging.debug(f"File tag: {file_tag}")
            if file_tag is None:
                logging.debug("File tag not found")
                continue
            batch_info: PurePath = file_tag.parent / config.BATCH_INFO
            logging.debug(f"Batch info: {batch_info}")
            try:
                batch_info_content: str = (await self.read_files(cast(PurePosixPath, batch_info)))[0]
            except FileNotFoundError:
                batch_info_content: str = "1: " + str(file_tag.parent / config.NANOPARTICLE_IN)
            file_read_output, files_to_read = await self._read_required_files(batch_info_content)
            total_steps: int = 0
            count: int = 0
            for folder, step, title in self._get_execution_data(file_read_output, files_to_read):
                total_steps += step
                count += 1
                if title != config.FINISHED_JOB:
                    yield LiveExecution(title, step, folder)
            if batch_info_content.count("\n") > 1:
                logging.debug(f"Found Batch execution")
                yield LiveExecution(config.BATCH_EXECUTION + f" ({count})", total_steps, PurePath(file_tag).parent)
        if connected:
            await self.disconnect()

    def _get_execution_data(self, file_read_output: dict[str, str | None], files_to_read: list[tuple[str, str, str]]):
        for folder_name, lammps_log, remote_nano_in in files_to_read:
            current_step: int = -1
            title: str = remote_nano_in
            lammps_log_content: str | None = file_read_output[lammps_log]
            if lammps_log_content is not None:
                try:
                    if "Total wall time" in lammps_log_content:
                        yield folder_name, config.FULL_RUN_DURATION, config.FINISHED_JOB
                        continue
                    current_step = LammpsRun.compute_current_step(lammps_log_content)
                    logging.debug(f"Current step: {current_step} {folder_name}")
                except subprocess.CalledProcessError:
                    logging.debug(f"Error getting current step for {folder_name}")
            code: str | None = file_read_output[remote_nano_in]
            if code is None:
                logging.error(f"Could not read {remote_nano_in}")
            else:
                title = code.split("\n")[0][1:].strip()
            yield folder_name, current_step, title

    def _process_files_to_read(self, batch_info_content) -> Generator[tuple[str, str, str], None, None]:
        for line in batch_info_content.split("\n"):
            if line.strip() == "":
                continue
            index, nano_in = line.split(": ")
            folder = PurePath(nano_in.split(" # ")[0] + "/")
            logging.debug(f"Found folder: {folder}")
            f_name: str = folder.name
            folder_name: PurePath = config.TOKO_EXECUTION_PATH / f_name
            lammps_log: PurePath = folder_name / "log.lammps"
            remote_nano_in: PurePath = folder_name / config.NANOPARTICLE_IN
            yield folder_name, lammps_log, remote_nano_in

    async def _read_required_files(self, batch_info_content) -> tuple[dict[str, str | None], list[tuple[str, str, str]]]:
        files_to_read: list[tuple[str, str, str]] = list(self._process_files_to_read(batch_info_content))
        file_read_output: dict[str, str | None] = {}
        file_names_to_read = [*[x[1] for x in files_to_read], *[x[2] for x in files_to_read]]
        file_contents: list[str | None] = list(await self.read_files(*file_names_to_read))
        logging.info(f"Read {len(file_contents)} files")
        for file_name, content in zip(file_names_to_read, file_contents):
            file_read_output[file_name] = content
        return file_read_output, files_to_read

    async def read_files(self, *paths: PurePosixPath) -> list[str]:
        local: Path = Path("/") / "tmp" / f"dw_{random.randint(0, 10000)}"
        while local.exists():
            local: Path = Path("/") / "tmp" / f"dw_{random.randint(0, 10000)}"
        local.mkdir(parents=True, exist_ok=False)
        # await self.sftp.mget([str(path) for path in paths], str(local), error_handler=lambda x: 0)
        await asyncio.gather(*[self.cp_get(remote_path=path, local_path=local / path.parent.name, ignore_errors=True) for path in paths])
        logging.info(f"Read {len(paths)} files")
        out: list[str] = []
        for path in paths:
            out.append(utils.read_local_file(local / path.parent.name))
        shutil.rmtree(local)
        logging.info(f"Loaded {len(out)} files")
        return out
