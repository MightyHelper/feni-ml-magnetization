from model.machine import Machine
from simulation_task import SimulationTask


class SchedulerService:
    @staticmethod
    def schedule(machines: list[Machine], tasks: list[SimulationTask]) -> list[Machine]:
        """
        Given N machines, with [a, b, ..., z] cores, and M tasks.
        Assign tasks to cores in such a way that the total execution time is minimized.
        """
        queues: list[list[SimulationTask]] = [[] for machine in machines for _ in range(machine.cores)]

        queue_index: int = 0
        for task in tasks:
            queues[queue_index].append(task)
            queue_index = (queue_index + 1) % len(queues)

        for machine in machines:
            machine.task_queue = queues[:machine.cores]
            queues = queues[machine.cores:]

        return machines
