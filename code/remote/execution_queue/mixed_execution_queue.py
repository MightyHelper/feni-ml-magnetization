import logging
from multiprocessing.pool import Pool

from lammps.simulation_task import SimulationTask
from remote.execution_queue.execution_queue import ExecutionQueue
from remote.execution_queue.slurm_execution_queue import estimate_minutes, minutes_to_slurm
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

    def schedule(self, is_test: bool = False):
        result: tuple[list[list[SimulationTask]], float] = SchedulerService.schedule_queue(self.queues, self.queue, is_test)
        for execution_queue, queue in zip(self.queues, result[0]):
            queue: list[SimulationTask]
            for item in queue:
                self.queues[self.queues.index(execution_queue)].enqueue(item)

    def run(self) -> list[SimulationTask]:
        is_test_run: bool = False
        if len(self.queue) > 0:
            is_test_run = self.queue[0].is_test_run
        self.schedule(is_test_run)
        render_queue_plan(self, is_test_run)
        with Pool(len(self.queues)) as p:
            results = p.map(_run_queue, self.queues)
        return [item for sublist in results for item in sublist]


def render_queue_plan(queue: ExecutionQueue, is_test: bool = False, tolerance: float = 2.5) -> list[float]:
    if isinstance(queue, MixedExecutionQueue):
        return [qmin for q in queue.queues for qmin in render_queue_plan(q, is_test, tolerance)]
    else:
        qlen = len(queue.queue)
        qcores = queue.parallelism_count
        qperf = queue.remote.single_core_completion_time
        qmin = estimate_minutes(qlen, qcores, qperf, queue.remote.launch_time, is_test)
        qtime = minutes_to_slurm(qmin, tolerance)
        logging.warning(f"{queue.remote.name:12} ({queue.parallelism_count:3} cores): {len(queue.queue):5} tasks = {qtime}")
        return [qmin]
