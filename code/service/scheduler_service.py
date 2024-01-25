from remote.machine.machine import Machine
from lammps.simulation_task import SimulationTask


class SchedulerService:
    @staticmethod
    def schedule(machines: dict[str, Machine], tasks: list[SimulationTask]) -> tuple[list[Machine], int]:
        """
        Given N machines, with [a, b, ..., z] cores, and M tasks.
        Assign tasks to cores in such a way that the total execution time is minimized.
        """
        # Sort machines by number of cores
        mach: list[Machine] = list(machines.values())
        mach.sort(key=lambda machine: machine.single_core_performance, reverse=True)
        queues: list[list[SimulationTask]] = [[] for machine in mach for _ in range(machine.cores)]

        queue_index: int = 0
        for task in tasks:
            queues[queue_index].append(task)
            queue_index = (queue_index + 1) % len(queues)

        for machine in mach:
            machine.task_queue = queues[:machine.cores]
            queues = queues[machine.cores:]

        longest_queue: int = max([len(queue) for machine in mach for queue in machine.task_queue])

        return mach, longest_queue
