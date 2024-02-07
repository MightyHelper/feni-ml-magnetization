import logging
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, AsyncGenerator

import utils
from config import config
from lammps.nanoparticle import Nanoparticle
from model.live_execution import LiveExecution
from remote.machine.machine import Machine


@dataclass
class LocalMachine(Machine):
    def __init__(self, execution_path: Path, lammps_executable: Path, launch_time: float = 0.0, single_core_completion_time: float = 1.0):
        super().__init__("local", multiprocessing.cpu_count(), execution_path, lammps_executable, launch_time, single_core_completion_time)
        if platform.system() == "Windows":
            import win32file
            win32file._setmaxstdio(2048)
        elif platform.system() == "Linux":
            import resource
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))


    def run_cmd(self, command: list[str]) -> bytes:
        logging.debug(f"Running {command=}")
        return subprocess.check_output(command)

    def mkdir(self, remote_path: Path):
        os.makedirs(remote_path, exist_ok=True)

    def cp_to(self, local_path: Path, remote_path: Path, is_folder: bool):
        if local_path == remote_path:
            return
        shutil.copy(local_path, remote_path)

    def cp_multi_to(self, local_paths: list[Path], remote_path: Path):
        for local_path in local_paths:
            self.cp_to(local_path, remote_path / local_path.name, is_folder=True)

    def cp_multi_from(self, remote_paths: list[Path], local_path: Path):
        for remote_path in remote_paths:
            self.cp_from(remote_path, local_path / remote_path.name, is_folder=True)

    def cp_from(self, remote_path: Path, local_path: Path, is_folder: bool):
        self.cp_to(remote_path, local_path, is_folder)

    def read_file(self, filename: str) -> str:
        return utils.read_local_file(filename)

    def read_multiple_files(self, filenames: list[str]) -> list[str]:
        return [self.read_file(filename) for filename in filenames]

    def rm(self, file_path: Path | str):
        os.remove(file_path)

    def ls(self, remote_dir: Path | str) -> list[str]:
        return os.listdir(remote_dir)

    def copy_alloy_files(self, local_sim_folder: Path, remote_sim_folder: Path):
        local_alloy_file: Path = utils.assert_type(Path, local_sim_folder.parent.parent) / "FeCuNi.eam.alloy"
        remote_alloy_file: Path = utils.assert_type(Path, remote_sim_folder.parent.parent) / "FeCuNi.eam.alloy"
        logging.info("Copying alloy files...")
        self.cp_to(local_alloy_file, remote_alloy_file, False)

    def remove_dir(self, remote_dir: Path | str):
        # os.rmtree(remote_dir)
        remote_dir = Path(remote_dir)
        relative_path = remote_dir.resolve().expanduser().relative_to(self.execution_path.parent)
        # if reltive path exists and no ValueError was thrown we can delete
        shutil.rmtree(remote_dir)

    async def get_running_tasks(self) -> AsyncGenerator[LiveExecution, None]:
        if platform.system() == "Windows":
            for item in self.get_running_windows(True):
                yield item
        elif platform.system() == "Linux":
            # try:
            #     for item in self.get_running_windows(False):
            #         yield item
            # except FileNotFoundError:
            #     pass
            for item in self.get_running_linux():
                yield item
        else:
            raise Exception(f"Unknown system: {platform.system()}")

    def get_running_linux(self) -> Generator[LiveExecution, None, None]:
        from lammps.nanoparticle import Nanoparticle
        ps_result = os.popen("ps -ef | grep " + str(self.lammps_executable)).readlines()
        for execution in {x for result in ps_result if (x := re.sub(".*?(-in (.*))?\n", "\\2", result)) != ""}:
            folder_name = Path(execution).parent
            try:
                nano = Nanoparticle.from_executed(folder_name)
                yield LiveExecution(nano.title, nano.run.get_current_step(), folder_name)
            except Exception as e:
                logging.debug(f"Could not parse {folder_name} {e}")
                pass

    def get_running_windows(self, from_windows: bool = True) -> Generator[LiveExecution, None, None]:
        # wmic.exe process where "name='python.exe'" get commandline, disable stderr
        if from_windows:
            path = Path("C:\Windows\System32\wbem\wmic.exe")
        else:
            path = Path("/mnt/c/Windows/System32/Wbem/wmic.exe")
        result = self.run_cmd(
            [
                str(path),
                "process",
                "where",
                f"name='{config.LOCAL_LAMMPS_NAME_WINDOWS}'",
                "get",
                "commandline",
            ],
        ).decode('utf-8').replace("\r", "").split("\n")
        logging.debug(f"WMIC Result: {result}")
        result = [x.strip() for x in result if x.strip() != ""]
        xv: str = ""
        for execution in {
            xv
            for result in result
            if "-in" in result and (xv := re.sub(".*?(-in (.*))\n?", "\\2", result).strip()) != ""
        }:
            logging.debug(f"Found execution: {execution}")
            folder_name = Path(execution).parent
            nano = Nanoparticle.from_executed(folder_name)
            yield LiveExecution(nano.title, nano.run.get_current_step(), folder_name)

    def make_executable(self, local_path: Path):
        os.chmod(local_path, 0o755)
        logging.debug(f"Made {local_path} executable")
