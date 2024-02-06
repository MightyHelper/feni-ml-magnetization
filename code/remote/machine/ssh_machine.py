import asyncio
import logging
import random
import time
from asyncio import Task
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, AsyncGenerator, Coroutine, Iterable

import asyncssh
from asyncssh import SSHCompletedProcess, DISC_BY_APPLICATION, SFTPFailure

import utils
from config import config
from config.config import BATCH_INFO
from lammps.simulation_task import SimulationTask
from model.live_execution import LiveExecution
from remote.execution_queue.execution_queue import ExecutionQueue
from remote.machine.local_machine import LocalMachine
from remote.machine.machine import Machine
from template import TemplateUtils
from utils import set_type
from utils import write_local_file


@dataclass
class SSHMachine(Machine):
    """
    A remote machine with a number of cores that we can SSH into
    """
    user: str
    remote_url: str
    port: int
    password: str | None

    connection: asyncssh.SSHClientConnection | None = field(init=False, default=None)
    sftp: asyncssh.SFTPClient | None = field(init=False, default=None)

    semaphore = asyncio.Semaphore(200)

    def __init__(
        self,
        name: str,
        cores: int,
        user: str,
        remote_url: str,
        port: int,
        password: str | None = None,
        lammps_executable: PurePath = PurePath(),
        execution_path: PurePath = PurePath(),
        launch_time: float = 0.0,
    ):
        super().__init__(name, cores, execution_path, lammps_executable, launch_time)
        self.user = user
        self.remote_url = remote_url
        self.port = port
        self.password = password

    async def connect(self, start_sftp: bool = True):
        logging.info(f"Connecting to {self}...")
        self.connection = await asyncssh.connect(
            self.remote_url,
            port=self.port,
            username=self.user,
            password=self.password,
            client_host="foo",  # Otherwise getnameinfo might fail on our own address
        )
        self.connection.set_keepalive(interval=2)
        if start_sftp:
            self.sftp = await self.connection.start_sftp_client()

    async def get_running_tasks(self) -> AsyncGenerator[LiveExecution, None]:
        raise NotImplementedError("get_running_tasks not implemented in SSHMachine")

    def run_cmd(self, cmd: str) -> Task[SSHCompletedProcess]:
        return asyncio.create_task(self.connection.run(cmd))

    def cp_put(self, local_path: Path, remote_path: PurePosixPath) -> Coroutine[Any, Any, None]:
        logging.debug(f"Copying {local_path} to {remote_path}...")

        async def do_thing():
            async with self.semaphore:
                return await self.sftp.put([str(local_path.resolve())], str(remote_path), recurse=True)

        return do_thing()

    def cp_get(self, local_path: Path, remote_path: PurePosixPath) -> Coroutine[Any, Any, None]:
        logging.debug(f"Copying {remote_path} to {local_path}...")

        async def do_thing():
            async with self.semaphore:
                return await self.sftp.get([str(remote_path)], str(local_path.resolve()), recurse=True)

        return do_thing()

    def cp_nput(self, local_paths: Iterable[Path], remote_path: PurePosixPath) -> Coroutine[Any, Any, None]:
        logging.debug(f"Copying {local_paths} to {remote_path}...")

        async def do_thing():
            async with self.semaphore:
                return await self.sftp.put([str(local_path.resolve()) for local_path in local_paths], str(remote_path), recurse=True)

        return do_thing()

    def cp_nget(self, local_path: Path, remote_paths: Iterable[PurePosixPath]) -> Coroutine[Any, Any, None]:
        logging.debug(f"Copying {remote_paths} to {local_path}...")

        async def do_thing():
            async with self.semaphore:
                return await self.sftp.get([str(remote_path) for remote_path in remote_paths], str(local_path.resolve()), recurse=True)

        return do_thing()

    async def disconnect(self):
        logging.info(f"Disconnecting from {self}...")
        # We no longer do it like this because we could wait forever:
        # await self.sftp.wait_closed()
        # await self.connection.wait_closed()
        # Instead we do it like this:
        self.connection.disconnect(DISC_BY_APPLICATION, "Done Nya!")


class SSHBatchedExecutionQueue(ExecutionQueue):
    queue: list[SimulationTask]
    completed: list[SimulationTask]
    remote: SSHMachine
    local: LocalMachine

    def run(self) -> list[SimulationTask]:
        result: list[SimulationTask] = asyncio.run(self.main())
        logging.info(f"Completed {len(result)} tasks in {self}")
        return result

    async def main(self) -> list[SimulationTask]:
        try:
            await self.remote.connect()
            await self._simulate()
        except Exception as e:
            logging.error(f"Error in {type(self)}: {e}", stack_info=True, exc_info=e)
        finally:
            await self.remote.disconnect()
        return self.completed

    def __init__(self, remote: SSHMachine, local: LocalMachine, batch_size: int = 10):
        super().__init__()
        self.batch_size = batch_size
        self.remote = remote
        self.parallelism_count = batch_size
        self.local = local
        self.queue = []
        self.completed = []

    async def submit_remote_batch(self, batch_name: str):
        logging.info(f"Queueing job in {self}...")
        remote_batch_path: PurePosixPath = self._get_remote_exec_child(batch_name)
        return await self.remote.run_cmd(f"cd {remote_batch_path}; sh {config.RUN_SH}")

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
        self.local.make_executable(local_run_script_path)

    async def _simulate(self):
        logging.info(f"Simulating {len(self.queue)} tasks in {self}...")
        batch_name: str = f"batch_{int(time.time())}_{random.randint(0, 1000)}"
        self._setup_local_simulation_files(batch_name, self.queue, self.batch_size)
        await self._copy_scripts_to_remote(self.queue, batch_name)
        await self.submit_remote_batch(batch_name)
        await self._copy_scripts_from_remote(self.queue, batch_name)
        self.process_output(self.queue)

    def _get_remote_exec_child(self, child: str | PurePath) -> PurePosixPath:
        return set_type(PurePosixPath, self.remote.execution_path) / child

    def _get_local_exec_child(self, child: str | PurePath) -> Path:
        return set_type(Path, self.local.execution_path) / child

    async def _copy_scripts_to_remote(self, simulations: list[SimulationTask],
                                      batch_name: str) -> tuple[Path, PurePath]:
        """
        Copy the local batch folder, simulation folders and scripts to the remote machine
        :param simulations: Simulations to run
        :param batch_name: Name of the batch
        """
        remote_batch_path: PurePosixPath = self._get_remote_exec_child(batch_name)
        local_batch_path: Path = self._get_local_exec_child(batch_name)
        folder_cps: list[Coroutine[Any, Any, None]] = [
            # Batch folder
            self._cp_put(local_batch_path, remote_batch_path),
            # Alloy files
            self._cp_put(
                set_type(Path, self.local.execution_path).parent / "FeCuNi.eam.alloy",
                set_type(PurePosixPath, self.remote.execution_path).parent
            ),
            # self._cp_nput(
            #     [simulation.local_cwd for simulation in simulations],
            #     set_type(PurePosixPath, self.remote.execution_path)
            # )
            *[
                self._cp_put(
                    simulation.local_cwd,
                    self._get_remote_exec_child(simulation.local_cwd.name)
                )
                for simulation in simulations
            ]
        ]
        try:
            await asyncio.gather(*folder_cps)
        except SFTPFailure as e:
            logging.error(f"Error in {type(self)}: {e}", stack_info=True, exc_info=e)
            raise Exception(f"Error copying files") from e
        return local_batch_path, remote_batch_path

    async def _copy_scripts_from_remote(self, simulations: list[SimulationTask], batch_name: str) -> tuple[Path, PurePath]:
        """
        Copy the local batch folder, simulation folders and scripts to the remote machine
        :param simulations: Simulations to run
        :param batch_name: Name of the batch
        """
        remote_batch_path: PurePosixPath = self._get_remote_exec_child(batch_name)
        local_batch_path: Path = self._get_local_exec_child(batch_name)
        folder_cps: list[Coroutine[Any, Any, None]] = [
            # Batch folder
            self._cp_get(local_batch_path, remote_batch_path),
            *[
                self._cp_get(simulation.local_cwd, self._get_remote_exec_child(simulation.local_cwd.name))
                for simulation in simulations
            ]
            # self._cp_nget(local_batch_path, [self._get_remote_exec_child(simulation.local_cwd.name) for simulation in simulations])
        ]
        await asyncio.gather(*folder_cps)
        return local_batch_path, remote_batch_path

    def _cp_put(self, local_path: Path, remote_path: PurePosixPath) -> Coroutine[Any, Any, None]:
        return self.remote.cp_put(local_path, remote_path)

    def _cp_get(self, local_path: Path, remote_path: PurePosixPath) -> Coroutine[Any, Any, None]:
        self.local.remove_dir(local_path)
        return self.remote.cp_get(local_path, remote_path)

    def _cp_nput(self, local_paths: list[Path], remote_path: PurePosixPath) -> Coroutine[Any, Any, None]:
        return self.remote.cp_nput(local_paths, remote_path)

    def _cp_nget(self, local_path: Path, remote_paths: list[PurePosixPath]) -> Coroutine[Any, Any, None]:
        for remote_path in remote_paths:
            self.local.remove_dir(local_path / remote_path.name)
        return self.remote.cp_nget(local_path, remote_paths)

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
