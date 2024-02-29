import logging
import platform
import re
import subprocess
from multiprocessing.pool import ThreadPool
from pathlib import PurePath

from remote.execution_queue.execution_queue import SingleExecutionQueue, ExecutionQueue
from remote.machine.local_machine import LocalMachine
from lammps.simulation_task import SimulationTask


class LocalExecutionQueue(SingleExecutionQueue):
    remote: LocalMachine

    def __init__(self, local_machine: LocalMachine):
        super().__init__(local_machine)
        self.remote = local_machine

    def _simulate(self, simulation_task: SimulationTask) -> tuple[SimulationTask, str]:
        lammps_executable: PurePath = self.remote.lammps_executable
        cmd = f"{simulation_task.mpi} {lammps_executable} {simulation_task.omp} {simulation_task.gpu} -in {simulation_task.local_input_file}"
        cmd = re.sub(r' +', " ", cmd).strip()
        logging.info(
            f"[bold blue]LocalExecutionQueue[/bold blue] Running [bold yellow]{cmd}[/bold yellow] in [cyan]{simulation_task.local_cwd}[/cyan]",
            extra={"markup": True, "highlighter": None}
        )
        try:
            result = subprocess.check_output(
                cmd.split(" "),
                cwd=simulation_task.local_cwd,
                shell=platform.system() == "Windows"
            )
            return simulation_task, result.decode("utf-8")
        except subprocess.CalledProcessError as e:
            simulation_task.ok = False
            self.print_error(e)
            raise e
        except OSError as e:
            simulation_task.ok = False
            self.print_error(e)
            raise ValueError(f"Is LAMMPS ({lammps_executable}) installed?") from e


def _run_queue(queue: ExecutionQueue) -> list[SimulationTask]:
    return queue.run()


class ThreadedLocalExecutionQueue(ExecutionQueue):
    remote: LocalMachine
    queue: list[SimulationTask]

    def __init__(self, remote: LocalMachine, threads: int):
        super().__init__()
        self.threads: int = threads
        self.parallelism_count = threads
        self.index: int = 0
        self.remote = remote
        self.queue = []
        self.queues: list[LocalExecutionQueue] = [LocalExecutionQueue(self.remote) for _ in range(threads)]
        self.full_queue: list[SimulationTask] = []
        self.full_count: int = 0
        self.full_completed_count: int = 0

    def enqueue(self, simulation_task: SimulationTask):
        queue_to_use: int = self.index % len(self.queues)
        self.queues[queue_to_use].enqueue(simulation_task)
        self.full_queue.append(simulation_task)
        self.index += 1
        self.queue.append(simulation_task)

    def sub_queue_progress(self, progress: int, total: int, task: tuple[SimulationTask, str | None], sender: LocalExecutionQueue):
        self.full_completed_count += 1
        self.dispatch_message(ExecutionQueue.PROGRESS, progress=self.full_completed_count, total=self.full_count, task=task)

    def run(self) -> list[SimulationTask]:
        self.full_count = len(self.full_queue)
        self.full_completed_count = 0
        for queue in self.queues:
            queue.listen(ExecutionQueue.PROGRESS, self.sub_queue_progress)
        with ThreadPool(len(self.queues)) as p:
            results = p.map(_run_queue, self.queues)
        return [task for queue_result in results for task in queue_result]

    def __str__(self):
        return f"{type(self).__name__}({len(self.queues)} queues, {[len(q.queue) for q in self.queues]} items [{sum([len(q.queue) for q in self.queues])} total])"
