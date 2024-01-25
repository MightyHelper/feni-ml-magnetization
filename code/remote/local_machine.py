import logging
import multiprocessing
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator

import utils
from model.live_execution import LiveExecution
from remote.machine import Machine


@dataclass
class LocalMachine(Machine):
    def __init__(self, execution_path: Path, lammps_executable: Path):
        super().__init__("local", multiprocessing.cpu_count(), execution_path, lammps_executable)

    def run_cmd(self, command_getter: Callable[[], list[str]]) -> bytes:
        command = command_getter()
        logging.debug(f"Running {command=}")
        return subprocess.check_output(command)

    def mkdir(self, remote_path: Path):
        os.makedirs(remote_path, exist_ok=True)

    def cp_to(self, local_path: Path, remote_path: Path, is_folder: bool):
        if local_path == remote_path:
            return
        if is_folder:
            self.run_cmd(lambda: ["cp", "-r", local_path, remote_path])
        else:
            self.run_cmd(lambda: ["cp", local_path, remote_path])

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

    def get_running_tasks(self) -> Generator[LiveExecution, None, None]:
        from nanoparticle import Nanoparticle
        ps_result = os.popen("ps -ef | grep " + self.lammps_executable.as_posix()).readlines()
        for execution in {x for result in ps_result if (x := re.sub(".*?(-in (.*))?\n", "\\2", result)) != ""}:
            folder_name = Path(execution).parent
            try:
                nano = Nanoparticle.from_executed(folder_name)
                yield folder_name, nano.run.get_current_step(), nano.title
            except Exception as e:
                logging.debug(f"Could not parse {folder_name} {e}")
                pass
