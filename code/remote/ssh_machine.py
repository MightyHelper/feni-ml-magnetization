import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from asyncio import Task
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from typing import Generator, Callable, TypeVar, Any

import asyncssh

import config
import utils
from config import BATCH_INFO
from execution_queue import ExecutionQueue
from model.live_execution import LiveExecution
from remote.local_machine import LocalMachine
from remote.machine import Machine
from simulation_task import SimulationTask
from template import TemplateUtils
from utils import set_type
from utils import write_local_file

T = TypeVar('T')


class CopyProvider(ABC):
    @abstractmethod
    def get_copy_to(self, local_paths: list[Path], remote_path: PurePosixPath, recursive: bool) -> Callable[
        [str, str, int], list[str]]:
        pass

    @abstractmethod
    def get_copy_from(self, remote_paths: list[PurePosixPath], local_path: Path, recursive: bool) -> Callable[
        [str, str, int], list[str]]:
        pass


class SCPCopyProvider(CopyProvider):
    def get_copy_from(self, remote_paths: list[PurePosixPath], local_path: Path, recursive: bool) -> Callable[
        [str, str, int], list[str]]:
        def gen_cmd(user: str, remote_url: str, remote_port: int):
            return [
                "scp",
                "-P",
                f"{remote_port}",
                "-r",
                *[f"{user}@{remote_url}:{remote_path}" for remote_path in remote_paths],
                local_path.as_posix()
            ]

        return gen_cmd

    def get_copy_to(self, local_paths: list[Path], remote_path: PurePosixPath, recursive: bool) -> Callable[
        [str, str, int], list[str]]:
        def gen_cmd(user: str, remote_url: str, remote_port: int):
            return [
                "scp",
                "-P",
                f"{remote_port}",
                "-r",
                *[str(local_path) for local_path in local_paths],
                f"{user}@{remote_url}:{remote_path}"
            ]

        return gen_cmd


class RsyncCopyProvider(CopyProvider):
    def get_copy_from(self, remote_paths: list[PurePosixPath], local_path: Path, recursive: bool) -> Callable[
        [str, str, int], list[str]]:
        def gen_cmd(user: str, remote_url: str, remote_port: int):
            return [
                "rsync",
                "-ar",
                *[f"rsync://{user}@{remote_url}:{remote_port}/{remote_path}" for remote_path in remote_paths],
                local_path.as_posix()
            ]

        return gen_cmd

    def get_copy_to(self, local_paths: list[Path], remote_path: PurePosixPath, recursive: bool) -> Callable[
        [str, str, int], list[str]]:
        def gen_cmd(user: str, remote_url: str, remote_port: int):
            return [
                "rsync",
                "-ar",
                *[str(local_path) for local_path in local_paths],
                f"rsync://{user}@{remote_url}:{remote_port}/{remote_path}"
            ]

        return gen_cmd


@dataclass
class SSHMachine(Machine):
    """
    A remote machine with a number of cores that we can SSH into
    """

    def mkdir(self, remote_path: PurePath):
        pass

    def cp_to(self, local_path: Path, remote_path: PurePath, is_folder: bool):
        pass

    def cp_multi_to(self, local_paths: list[Path], remote_path: PurePath):
        pass

    def cp_multi_from(self, remote_paths: list[PurePath], local_path: Path):
        pass

    def cp_from(self, remote_path: PurePath, local_path: Path, is_folder: bool):
        pass

    def read_file(self, filename: PurePath) -> str:
        pass

    def read_multiple_files(self, filenames: list[PurePath]) -> list[str]:
        pass

    def rm(self, file_path: PurePath):
        pass

    def ls(self, remote_dir: PurePath) -> list[str]:
        pass

    def remove_dir(self, remote_dir: PurePath):
        pass

    user: str
    remote_url: str
    port: int
    password: str | None
    copy_script: PurePath

    copy_provider: CopyProvider

    def __init__(
            self,
            name: str,
            cores: int,
            user: str,
            remote_url: str,
            port: int,
            password: str | None = None,
            copy_script: PurePath = PurePath("rsync"),
            lammps_executable: PurePath = PurePath(),
            execution_path: PurePath = PurePath()
    ):
        super().__init__(name, cores, execution_path, lammps_executable)
        self.user = user
        self.remote_url = remote_url
        self.port = port
        self.password = password
        self.copy_script = copy_script
        self.copy_provider = SCPCopyProvider() if copy_script == 'scp' else RsyncCopyProvider()

    def get_running_tasks(self) -> Generator[LiveExecution, None, None]:
        raise NotImplementedError("get_running_tasks not implemented in SSHMachine")

class SSHBatchedExecutionQueue(ExecutionQueue):
    queue: list[SimulationTask]
    completed: list[SimulationTask]
    remote: SSHMachine
    local: LocalMachine

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
        return asyncio.run(self.main())

    async def main(self) -> list[SimulationTask]:
        async with asyncssh.connect(
                self.remote.remote_url,
                port=self.remote.port,
                username=self.remote.user,
                password=self.remote.password,
                client_host="foo"  # Otherwise getnameinfo might fail on our own address
        ) as connection:
            async with connection.start_sftp_client() as sftp:
                try:
                    await self._simulate(connection, sftp)
                except Exception as e:
                    logging.error(f"Error in {type(self)}: {e}", stack_info=True, exc_info=e)
                return self.completed

    def __init__(self, remote: SSHMachine, local: LocalMachine, batch_size: int = 10):
        super().__init__()
        self.batch_size = batch_size
        self.remote = remote
        self.local = local
        self.queue = []
        self.completed = []

    async def submit_remote_batch(self, batch_name: str, connection: asyncssh.SSHClientConnection):
        logging.info(f"Queueing job in {self}...")
        remote_batch_path: PurePosixPath = self._get_remote_exec_child(batch_name)
        return await connection.run(f"cd {remote_batch_path}; sh {config.RUN_SH}")
        # return await connection.run(f"echo Hello World")

    def _generate_local_run_file(self, batch_name: str, n_threads: int, simulation_count: int):
        remote_batch_path: PurePosixPath = utils.set_type(PurePosixPath, self.remote.execution_path) / batch_name
        local_run_script_path: Path = self._get_local_exec_child(batch_name) / config.RUN_SH
        script_code: str = TemplateUtils.replace_templates(
            TemplateUtils.get_ssh_multi_template(), {
                "tasks": str(n_threads),
                "cmd_args": str(BATCH_INFO),
                "cwd": str(remote_batch_path),
                "output": str(remote_batch_path / "batch_run.out"),
                "file_tag": str(remote_batch_path / config.RUN_SH),
            }
        )
        assert "{{" not in script_code, f"Not all templates were replaced in {script_code} for {self}"
        write_local_file(local_run_script_path, script_code)
        # Change permission u+x
        self.local.run_cmd(lambda: ["chmod", "u+x", local_run_script_path])

    async def _simulate(self, connection: asyncssh.SSHClientConnection, sftp: asyncssh.SFTPClient):
        batch_name: str = f"batch_{int(time.time())}_{random.randint(0, 1000)}"
        self._setup_local_simulation_files(batch_name, self.queue, self.batch_size)
        await self._copy_scripts_to_remote(sftp, self.queue, batch_name)
        await self.submit_remote_batch(batch_name, connection)
        await self._copy_scripts_from_remote(sftp, self.queue, batch_name)
        self.process_output(self.queue)

    def _get_remote_exec_child(self, child: str | PurePath) -> PurePosixPath:
        return set_type(PurePosixPath, self.remote.execution_path) / child

    def _get_local_exec_child(self, child: str | PurePath) -> Path:
        return set_type(Path, self.local.execution_path) / child

    async def _copy_scripts_to_remote(self, sftp: asyncssh.SFTPClient, simulations: list[SimulationTask],
                                      batch_name: str) -> tuple[Path, PurePath]:
        """
        Copy the local batch folder, simulation folders and scripts to the remote machine
        :param sftp: SFTP client
        :param simulations: Simulations to run
        :param batch_name: Name of the batch
        """
        remote_batch_path: PurePosixPath = self._get_remote_exec_child(batch_name)
        local_batch_path: Path = self._get_local_exec_child(batch_name)
        folder_cps: list[Task] = [
            # Batch folder
            self._cp_put(sftp, local_batch_path, remote_batch_path),
            # Alloy files
            self._cp_put(
                sftp,
                set_type(Path, self.local.execution_path).parent / "FeCuNi.eam.alloy",
                set_type(PurePosixPath, self.remote.execution_path).parent
            ),
            *[
                self._cp_put(
                    sftp,
                    simulation.local_cwd,
                    self._get_remote_exec_child(simulation.local_cwd.name)
                )
                for simulation in simulations
            ]
        ]
        for folder_cp in folder_cps:
            await folder_cp
        return local_batch_path, remote_batch_path

    async def _copy_scripts_from_remote(self, sftp: asyncssh.SFTPClient, simulations: list[SimulationTask],
                                      batch_name: str) -> tuple[Path, PurePath]:
        """
        Copy the local batch folder, simulation folders and scripts to the remote machine
        :param sftp: SFTP client
        :param simulations: Simulations to run
        :param batch_name: Name of the batch
        """
        remote_batch_path: PurePosixPath = self._get_remote_exec_child(batch_name)
        local_batch_path: Path = self._get_local_exec_child(batch_name)
        folder_cps: list[Task] = [
            # Batch folder
            self._cp_get(sftp, local_batch_path, remote_batch_path),
            *[
                self._cp_get(
                    sftp,
                    simulation.local_cwd,
                    self._get_remote_exec_child(simulation.local_cwd.name)
                )
                for simulation in simulations
            ]
        ]
        for folder_cp in folder_cps:
            await folder_cp
        return local_batch_path, remote_batch_path

    def _cp_put(self, sftp: asyncssh.SFTPClient, local_path: Path, remote_path: PurePosixPath) -> Task:
        return asyncio.create_task(sftp.put([local_path.resolve().as_posix()], remote_path.as_posix(), recurse=True))

    def _cp_get(self, sftp: asyncssh.SFTPClient, local_path: Path, remote_path: PurePosixPath) -> Task:
        self.local.remove_dir(local_path)
        return asyncio.create_task(sftp.get([remote_path.as_posix()], local_path.resolve().as_posix(), recurse=True))

    def _setup_local_simulation_files(self, batch_name: str, simulations, n_threads: int):
        """
        Create a folder in local with the simulations and a multi.py file to run them in parallel
        :param simulations: Simulations to run
        :param n_threads: Number of threads to specify in the slurm file
        :return:
        """
        # Make new dir in remote execution path "batch_<timestamp>_<random[0-1000]>"
        local_batch_path: Path = self._get_local_exec_child(batch_name)
        self.local.mkdir(local_batch_path)
        self.local.cp_to(config.LOCAL_MULTI_PY, local_batch_path / config.LOCAL_MULTI_PY.name, False)
        self._generate_local_tasks_file(batch_name, simulations)
        self._generate_local_run_file(batch_name, n_threads, len(simulations))

    def _generate_local_tasks_file(self, batch_name: str, simulations: list[SimulationTask]):
        tasks: list[str] = []
        local_batch_path: Path = self._get_local_exec_child(batch_name)
        batch_info_path: Path = local_batch_path / BATCH_INFO
        for simulation in simulations:
            remote_sim_folder: PurePosixPath = self._get_remote_exec_child(simulation.local_cwd.name)
            remote_nano_in_path: PurePosixPath = remote_sim_folder / config.NANOPARTICLE_IN
            remote_lammps_log_path: PurePosixPath = remote_sim_folder / config.LOG_LAMMPS
            tasks.append(
                f"sh -c 'cd {remote_sim_folder}; {self.remote.lammps_executable} -in {remote_nano_in_path} > {remote_lammps_log_path}'")
        write_local_file(batch_info_path, "\n".join([
            f"{i + 1}: {Path(simulation.local_input_file).parent.resolve()} # {shell}"
            for i, (simulation, shell) in enumerate(zip(simulations, tasks))
        ]) + "\n")

    def process_output(self, simulations: list[SimulationTask]):
        logging.info("Copying output files from remote to local machine...")
        callback_info: list[tuple[Any, Path]] = []
        for i, simulation in enumerate(simulations):
            local_sim_folder: Path = Path(simulation.local_input_file).parent.resolve()
            local_lammps_log: Path = local_sim_folder / "log.lammps"
            callback_info.append((simulation, local_lammps_log))
        for simulation, lammps_log in callback_info:
            self.run_callback(simulation, utils.read_local_file(lammps_log))
            self.completed.append(simulation)
