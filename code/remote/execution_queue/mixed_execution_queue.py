from multiprocessing.pool import ThreadPool, Pool
from lammps.simulation_task import SimulationTask
from remote.execution_queue.execution_queue import ExecutionQueue
from remote.machine.machine import Machine
from service.scheduler_service import SchedulerService


def _run_queue(queue: ExecutionQueue) -> list[SimulationTask]:
    return queue.run()


class MixedExecutionQueue(ExecutionQueue):
    queue: list[SimulationTask]
    queues: list[ExecutionQueue]

    def __init__(self, queues: list[ExecutionQueue]):
        super().__init__()
        self.queue = []
        self.queues = queues

    def enqueue(self, simulation_task: SimulationTask):
        self.queue.append(simulation_task)

    def schedule(self):
        result: tuple[list[list[SimulationTask]], float] = SchedulerService.schedule_queue(self.queues, self.queue)
        for execution_queue, queue in zip(self.queues, result[0]):
            queue: list[SimulationTask]
            for item in queue:
                self.queues[self.queues.index(execution_queue)].enqueue(item)

    def run(self) -> list[SimulationTask]:
        self.schedule()
        with Pool(len(self.queues)) as p:
            results = p.map(_run_queue, self.queues)
        return [item for sublist in results for item in sublist]
