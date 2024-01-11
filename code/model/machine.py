from dataclasses import dataclass, field
from typing import Annotated

from simulation_task import SimulationTask


@dataclass
class Machine:
    """
    A machine with a number of cores
    """
    name: str
    cores: int
    single_core_performance: float = field(init=False, default=1.0)

    task_queue: list[list[SimulationTask]] = field(init=False, default_factory=list)

    def __str__(self) -> str:
        return f"Machine: {self.name} ({self.cores} cores) with {sum([len(queue) for queue in self.task_queue])} tasks"

    def __repr__(self) -> str:
        return str(self)


@dataclass
class SSHMachine(Machine):
    """
    A remote machine with a number of cores that we can SSH into
    """
    hostname: str
    user: str


@dataclass
class SLURMMachine(SSHMachine):
    """
    A remote machine with a number of cores that we can SSH into and use SLURM
    """
    partition: str
    node_id: int

    def __str__(self) -> str:
        return f"Machine: {self.name}/{self.partition}{self.node_id} ({self.cores} cores) with {sum([len(queue) for queue in self.task_queue])} tasks"


