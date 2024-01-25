from math import ceil

from remote.machine.machine import Machine
from lammps.simulation_task import SimulationTask


class SchedulerService:
    @staticmethod
    def estimate_queue_time(machine: Machine, tasks: list[SimulationTask]) -> float:
        """
        Estimate the time it takes to run a list of tasks on a machine
        """
        task_count = len(tasks)
        if task_count == 0:
            return 0
        return ceil(task_count / machine.cores) / machine.single_core_performance

    @staticmethod
    def schedule(machines: list[Machine], tasks: list[SimulationTask]) -> tuple[list[list[SimulationTask]], float]:
        """
        Given N machines, with [a, b, ..., z] cores, and M tasks.
        Assign tasks to cores in such a way that the total execution time is minimized.
        """
        queues: list[list[SimulationTask]] = [[] for _ in machines]
        for task in tasks:
            # best_queue: Machine = min(queues, key=lambda queue: SchedulerService.estimate_queue_time(machines[queue], queues[queue]))
            best_queue: tuple[Machine, list[SimulationTask]] = min(zip(machines, queues), key=lambda m_q: SchedulerService.estimate_queue_time(m_q[0], m_q[1]))
            best_queue[1].append(task)

        return queues, max([SchedulerService.estimate_queue_time(machine, queue) for machine, queue in zip(machines, queues)])

