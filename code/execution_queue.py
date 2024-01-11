import logging
import platform
import re
import subprocess
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool

from config_base import LAMMPS_EXECUTABLE
from simulation_task import SimulationTask


class ExecutionQueue(ABC):
    @abstractmethod
    def enqueue(self, simulation_task: SimulationTask):
        """
        Enqueue a simulation
        :param simulation_task: Simulation task to enqueue
        :return:
        """
        pass

    @abstractmethod
    def run(self) -> list[SimulationTask]:
        """
        Run all simulation tasks
        :return:
        """
        pass

    def run_callback(self, simulation_task: SimulationTask, result: str | None):
        for callback in simulation_task.callbacks:
            try:
                callback(result)
            except Exception as e:
                logging.warning(f"Error in run_callback ({callback})", exc_info=e)

    def __str__(self):
        return f"{type(self).__name__}"

    def __repr__(self):
        return self.__str__()


class SingleExecutionQueue(ExecutionQueue, ABC):
    queue: list[SimulationTask]
    completed: list[SimulationTask]

    def __init__(self):
        super().__init__()
        self.queue = []
        self.completed = []

    def enqueue(self, simulation_task: SimulationTask):
        assert isinstance(simulation_task, SimulationTask)
        assert simulation_task.input_file is not None
        assert simulation_task.cwd is not None
        assert simulation_task.mpi is not None
        assert simulation_task.omp is not None
        assert simulation_task.gpu is not None
        assert simulation_task not in self.queue
        self.queue.append(simulation_task)

    def _get_next_task(self) -> SimulationTask:
        if len(self.queue) == 0:
            return None
        element = self.queue[0]
        self.queue = self.queue[1:]
        return element

    def print_error(self, e, **kwargs):
        kwargs['queue'] = type(self)
        kwargs['queue'] = len(self.queue)
        params = "\n".join([f"{key}={value}" for key, value in kwargs.items()]) if kwargs else ""
        logging.error("ERROR:" + str(e) + " " + params, extra={"markup": True})

    def run(self) -> list[SimulationTask]:
        while len(self.queue) > 0:
            task: SimulationTask | None = self._get_next_task()
            result: tuple[SimulationTask, str | None] = (task, None)
            if task is None:
                break
            try:
                result = self._simulate(task)
            except Exception as e:
                logging.error(f"Error in {type(self)}: {e}", stack_info=False)
            finally:
                self.completed.append(result[0])
                self.run_callback(result[0], result[1])
        return self.completed

    @abstractmethod
    def _simulate(self, simulation_task: SimulationTask) -> tuple[SimulationTask, str]:
        pass

    def __str__(self):
        return f"{type(self).__name__}({len(self.queue)} items)"


class LocalExecutionQueue(SingleExecutionQueue):
    def _simulate(self, simulation_task: SimulationTask) -> tuple[SimulationTask, str]:
        lammps_executable = LAMMPS_EXECUTABLE
        cmd = f"{simulation_task.mpi} {lammps_executable} {simulation_task.omp} {simulation_task.gpu} -in {simulation_task.input_file}"
        cmd = re.sub(r' +', " ", cmd).strip()
        logging.info(
            f"[bold blue]LocalExecutionQueue[/bold blue] Running [bold yellow]{cmd}[/bold yellow] in [cyan]{simulation_task.cwd}[/cyan]",
            extra={"markup": True, "highlighter": None})
        try:
            result = subprocess.check_output(cmd.split(" "), cwd=simulation_task.cwd,
                                             shell=platform.system() == "Windows")
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
    def __init__(self, threads: int):
        super().__init__()
        self.threads: int = threads
        self.index: int = 0
        self.queues: list[LocalExecutionQueue] = [LocalExecutionQueue() for _ in range(threads)]
        self.full_queue: list[SimulationTask] = []

    def enqueue(self, simulation_task: SimulationTask):
        queue_to_use: int = self.index % len(self.queues)
        self.queues[queue_to_use].enqueue(simulation_task)
        self.full_queue.append(simulation_task)
        self.index += 1

    def run(self) -> list[SimulationTask]:
        with ThreadPool(len(self.queues)) as p:
            results = p.map(_run_queue, self.queues)
        return [task for queue_result in results for task in queue_result]

    def __str__(self):
        return f"{type(self).__name__}({len(self.queues)} queues, {[len(q.queue) for q in self.queues]} items [{sum([len(q.queue) for q in self.queues])} total])"
