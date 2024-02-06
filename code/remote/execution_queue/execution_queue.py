import logging
from abc import ABC, abstractmethod
from lammps.simulation_task import SimulationTask
from remote.machine.machine import Machine


class ExecutionQueue(ABC):
    queue: list[SimulationTask]
    parallelism_count: int
    remote: Machine

    def enqueue(self, simulation_task: SimulationTask):
        """
        Enqueue a simulation
        :param simulation_task: Simulation task to enqueue
        :return:
        """
        assert isinstance(simulation_task, SimulationTask)
        assert simulation_task.local_input_file is not None
        assert simulation_task.local_cwd is not None
        assert simulation_task.mpi is not None
        assert simulation_task.omp is not None
        assert simulation_task.gpu is not None
        assert simulation_task not in self.queue
        self.queue.append(simulation_task)

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

    def __init__(self, remote: Machine):
        super().__init__()
        self.remote = remote
        self.queue = []
        self.parallelism_count = 1
        self.completed = []

    def _get_next_task(self) -> SimulationTask | None:
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
                logging.error(f"Error in {type(self)}: {e}")
                logging.debug(f"Error in {type(self)}: {e}", exc_info=e, stack_info=True)
            finally:
                self.completed.append(result[0])
                self.run_callback(result[0], result[1])
        return self.completed

    @abstractmethod
    def _simulate(self, simulation_task: SimulationTask) -> tuple[SimulationTask, str]:
        pass

    def __str__(self):
        return f"{type(self).__name__}({len(self.queue)} items)"
