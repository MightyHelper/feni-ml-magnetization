import logging
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from typing import Generator, cast

import config
import utils
from config import BATCH_INFO_PATH, LOCAL_EXECUTION_PATH
from execution_queue import ExecutionQueue
from model.live_execution import LiveExecution
from remote.local_machine import LocalMachine
from remote.machine import Machine
from simulation_task import SimulationTask
from template import TemplateUtils
from utils import write_local_file


@dataclass
class SSHMachine(Machine):
    """
    A remote machine with a number of cores that we can SSH into
    """

    user: str
    remote_url: str
    copy_script: PurePath

    def __init__(
            self,
            name: str,
            cores: int,
            user: str,
            remote_url: str,
            copy_script: PurePath = 'rsync',
            lammps_executable: PurePath = PurePath('lmp'),
            execution_path: PurePath = PurePath('~/magnetism/simulations')
    ):
        super().__init__(name, cores, execution_path, lammps_executable)
        self.user = user
        self.remote_url = remote_url
        self.copy_script = copy_script

    def run_cmd(self, command_getter) -> bytes:
        command = command_getter(self.user, self.remote_url)
        logging.debug(f"Running {command=}")
        return subprocess.check_output(command)

    def mkdir(self, remote_path: PurePosixPath) -> bytes:
        return self.run_cmd(
            lambda user, remote_url: [
                "ssh",
                f"{user}@{remote_url}",
                f"mkdir",
                f"{remote_path.as_posix()}"
            ]
        )

    def cp_to(self, local_path: Path, remote_path: PurePosixPath, is_folder: bool = False) -> bytes:
        # Convert Potential CRLF to LF
        local_path_posix: str = ""
        remote_path_posix: str = ""
        utils.assert_type(Path, local_path)
        utils.assert_type(PurePosixPath, remote_path)
        if not is_folder:
            self.convert_crlf_to_lf(local_path)
        if self.copy_script == 'scp':
            remote_folder = remote_path.parent
            local_folder = local_path.parent
            assert remote_folder == local_folder, f"remote_folder: {remote_folder}, local_folder: {local_folder}"
            remote_path = remote_path.name
        local_path_posix = local_path.as_posix()
        remote_path_posix = remote_path.as_posix()
        if is_folder and not local_path_posix.endswith("/"):
            local_path_posix = local_path_posix + "/"
        return self.run_cmd(
            lambda user, remote_url:
            [self.copy_script, "-r", local_path_posix, f"{user}@{remote_url}:{remote_path_posix}/"] if is_folder else
            [self.copy_script, local_path_posix, f"{user}@{remote_url}:{remote_path_posix}"],
        )

    def convert_crlf_to_lf(self, local_path: Path):
        with open(local_path, 'rb') as f:
            content = f.read()
        content = content.replace(b'\r\n', b'\n')
        with open(local_path, 'wb') as f:
            f.write(content)

    def cp_multi_to(self, local_paths: list[Path], remote_path: PurePosixPath) -> str:
        logging.info(f"Copying {len(local_paths)} files to remote {remote_path}...")
        max_len: int = 8000  # 8191 official limit
        batches: list[list[str]] = []
        batch: str = ""
        batch_list: list[str] = []
        for local_path in local_paths:
            local_path_posix: str = local_path.as_posix()
            if len(batch) + len(local_path_posix) + 1 > max_len:
                batches.append(batch_list)
                batch_list = []
                batch = ""
            batch += local_path_posix + " "
            batch_list.append(local_path_posix)
        if len(batch_list) > 0:
            batches.append(batch_list)
        cmd_out = ""
        for b in batches:
            cmd_out += self.run_cmd(
                lambda user, remote_url: [
                    self.copy_script,
                    "-ar" if self.copy_script == 'rsync' else '-r',
                    *b,
                    f"{user}@{remote_url}:{remote_path}"
                ]).decode("utf-8")
        return cmd_out

    def cp_multi_from(self, remote_paths: list[str], local_path: str) -> bytes:
        logging.info(f"Copying {len(remote_paths)} files from remote to local {local_path}...")
        return self.run_cmd(
            lambda user, remote_url: [
                self.copy_script,
                "-ar" if self.copy_script == 'rsync' else '-r',
                *[f"{user}@{remote_url}:{remote_path}" for remote_path in remote_paths],
                local_path
            ]
        )

    def cp_from(self, remote_path: str, local_path: str, is_folder: bool = False) -> bytes:
        return self.run_cmd(
            lambda user, remote_url:
            [self.copy_script, "-r", f"{user}@{remote_url}:{remote_path}/*", local_path] if is_folder else
            [self.copy_script, f"{user}@{remote_url}:{remote_path}", local_path]
        )

    def read_file(self, filename: PurePosixPath) -> str:
        """
        Read a file in remote
        :param filename:
        :return:
        """
        try:
            return self.run_cmd(lambda user, remote_url: [
                "ssh",
                f"{user}@{remote_url}",
                f"cat {filename.as_posix()}"
            ]).decode("utf-8")
        except subprocess.CalledProcessError:
            raise FileNotFoundError("TOKO: " + filename.as_posix())

    def read_multiple_files(self, filenames: list[PurePosixPath]) -> list[str]:
        """
        Read multiple files in remote
        :param filenames:
        :return:
        """
        # Temp separate files by a DELIMITER, and then split
        max_len: int = 8000  # 8191 official limit
        delimiter = "\n================\n"
        batches: list[list[str]] = []
        batch: str = ""
        batch_list: list[str] = []
        for filename in filenames:
            filename_posix: str = filename.as_posix()
            if len(batch) + len(filename_posix) + 1 > max_len:
                batches.append(batch_list)
                batch_list = []
                batch = ""
            batch += filename_posix + delimiter + "cat $ ()\"\""
            batch_list.append(filename_posix)
        if len(batch_list) > 0:
            batches.append(batch_list)
        out: list[str] = []
        for b in batches:
            logging.debug(f"Reading {len(b)} files from remote...")
            command = "echo -e \"" + delimiter.join([f"$(cat {filename})" for filename in b]) + "\""
            result: list[str] = self.run_cmd(
                lambda user, remote_url: [
                    "ssh",
                    f"{user}@{remote_url}",
                    f"sh -c '{command}' 2> /dev/null"
                ]).decode("utf-8").split(delimiter)
            out.extend(result)
            logging.debug(f"Read {len(result)} files from remote")
        return out

    def rm(self, *file_paths: PurePosixPath) -> bytes:
        return self.run_cmd(
            lambda user, remote_url: [
                "ssh",
                f"{user}@{remote_url}",
                f"rm {' '.join([path.as_posix() for path in file_paths])}"
            ]
        )

    def ls(self, remote_dir: PurePosixPath) -> list[str]:
        try:
            return self.run_cmd(
                lambda user, remote_url: [
                    "ssh",
                    f"{user}@{remote_url}",
                    f"ls {remote_dir.as_posix()}"
                ]
            ).decode("utf-8").strip().split("\n")
        except subprocess.CalledProcessError:
            raise FileNotFoundError("TOKO: " + remote_dir.as_posix())

    def remove_dir(self, remote_dir: PurePosixPath) -> bytes:
        return self.run_cmd(
            lambda user, remote_url: [
                "ssh",
                f"{user}@{remote_url}",
                f"rmdir {remote_dir.as_posix()}"
            ]
        )

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
        try:
            self._simulate()
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

    def submit_remote_batch(
            self,
            local_batch_path: Path,
            remote_batch_path: PurePath,
            n_threads: int = 1,
    ):
        # Create a slurm script to run the commands in parallel
        local_run_script_path = local_batch_path / config.RUN_SH
        remote_run_script_path = remote_batch_path / config.RUN_SH

        script_code = TemplateUtils.replace_templates(
            TemplateUtils.get_ssh_multi_template(), {
                "tasks": str(n_threads),
                "cmd_args": BATCH_INFO_PATH,
                "cwd": remote_batch_path,
                "output": remote_batch_path / "batch_run.out",
                "file_tag": remote_run_script_path
            }
        )
        assert "{{" not in script_code, f"Not all templates were replaced in {script_code} for {self}"
        write_local_file(local_run_script_path, script_code)
        logging.info(f"Copying run script to {self}...")
        self.remote.cp_to(local_run_script_path, remote_run_script_path)
        logging.info(f"Queueing job in {self}...")
        return self.remote.run_cmd(lambda user, remote_url: [
            "ssh",
            f"{user}@{remote_url}",
            f"sh -c 'cd {remote_batch_path}; sh {config.RUN_SH}'"
        ])

    def get_tasks(self, simulation_info):
        # Create a list of commands to run in parallel
        for remote_nano_in_path, remote_sim_folder, local_sim_folder in simulation_info:
            remote_lammps_log_path = remote_sim_folder / "log.lammps"
            yield f"sh -c 'cd {remote_sim_folder}; {self.remote.lammps_executable} -in {remote_nano_in_path} > {remote_lammps_log_path}'"

    def _simulate(self):
        simulations = self.queue
        local_batch_path, remote_batch_path, simulation_info = self.prepare_scripts(simulations)
        self.submit_remote_batch(
            local_batch_path,
            remote_batch_path,
            n_threads=self.batch_size,
        )
        self.process_output(local_batch_path, simulations, remote_batch_path)

    def prepare_scripts(self, simulations: list[SimulationTask]) -> tuple[Path, PurePath, list[tuple[str, str, str]]]:
        """
        Create a folder in remote with the simulations and a multi.py file to run them in parallel
        Then, copy the folder to remote
        :param simulations: List of simulations to run
        :return:
        """
        simulation_info = []
        # Make new dir in remote execution path "batch_<timestamp>_<random[0-1000]>"
        batch_name: Path = Path(f"batch_{int(time.time())}_{random.randint(0, 1000)}")
        local_batch_path: Path = LOCAL_EXECUTION_PATH / batch_name
        remote_batch_path: PurePosixPath = cast(PurePosixPath, self.remote.execution_path) / batch_name
        batch_info_path: Path = local_batch_path / BATCH_INFO_PATH
        local_multi_py: Path = config.LOCAL_MULTI_PY
        remote_multi_py: PurePosixPath = remote_batch_path / "multi.py"
        self.local.mkdir(local_batch_path)

        for i, simulation in enumerate(simulations):
            local_sim_folder: Path = simulation.local_input_file
            remote_sim_folder: PurePath = self.remote.execution_path / local_sim_folder.name
            remote_nano_in: PurePath = remote_sim_folder / Path(simulation.local_input_file).name
            logging.debug(f"Creating simulation folder in remote ({i + 1}/{len(simulations)})...")
            if i == 0:
                logging.debug(f"{remote_sim_folder=}")
                logging.debug(f"{local_sim_folder=}")
                logging.debug(f"{remote_nano_in=}")
                self.remote.copy_alloy_files(local_sim_folder, remote_sim_folder)
            simulation_info.append(
                (remote_nano_in.as_posix(), remote_sim_folder.as_posix(), local_sim_folder.as_posix()))
        tasks = self.get_tasks(simulation_info)
        write_local_file(batch_info_path, "\n".join([
            f"{i + 1}: {Path(simulation.local_input_file).parent.resolve()} # {shell}"
            for i, (simulation, shell) in enumerate(zip(simulations, tasks))
        ]) + "\n")

        self.remote.cp_multi_to([folder for (_, _, folder) in simulation_info], cast(PurePosixPath, self.remote.execution_path))

        self.remote.cp_to(local_batch_path, remote_batch_path, is_folder=True)
        self.remote.cp_to(local_multi_py, remote_multi_py)
        return local_batch_path, remote_batch_path, simulation_info

    def process_output(self, local_batch_path, simulations, remote_batch_path):
        logging.info("Copying output files from remote to local machine...")
        self.remote.cp_from(remote_batch_path, local_batch_path, is_folder=True)
        files_to_copy_back = []
        callback_info = []
        for i, simulation in enumerate(simulations):
            # Copy each simulation output to the local machine
            local_sim_folder: Path = Path(simulation.local_input_file).parent.resolve()
            remote_sim_path: PurePath = self.remote.execution_path / local_sim_folder.name
            local_lammps_log: Path = local_sim_folder / "log.lammps"
            files_to_copy_back.append(remote_sim_path.as_posix())
            callback_info.append((simulation, local_lammps_log))
        self.remote.cp_multi_from([folder for folder in files_to_copy_back], LOCAL_EXECUTION_PATH)
        for simulation, lammps_log in callback_info:
            self.run_callback(simulation, utils.read_local_file(lammps_log))
            self.completed.append(simulation)
