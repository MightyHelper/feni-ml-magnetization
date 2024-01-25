from pathlib import Path

import typer
from rich import print as rprint

from config import config
from lammps.nanoparticle import Nanoparticle
from remote.machine.machine import Machine
from remote.execution_queue.slurm_execution_queue import estimate_time
from service import executor_service
from service.scheduler_service import SchedulerService
from lammps.simulation_task import SimulationTask

sched = typer.Typer(add_completion=False, no_args_is_help=True, name="sched")


@sched.command()
def schedule(
        seed: int = typer.Option(
            0,
            help="Seed to start with",
            show_default=True
        ),
        seed_count: int = typer.Option(
            1,
            help="Number of seeds to run",
            show_default=True
        )
):
    """
    Schedule nanoparticle executions
    """
    nanoparticles: list[tuple[str, Nanoparticle]] = executor_service.build_nanoparticles_to_execute(
        [],
        Path("../Shapes"),
        seed,
        seed_count
    )
    tasks: list[SimulationTask] = [nanoparticle.get_simulation_task() for _, nanoparticle in nanoparticles]

    execution_plan: tuple[list[Machine], int] = SchedulerService.schedule(
        machines=config.MACHINES(),
        tasks=tasks
    )
    machines, longest_queue = execution_plan
    for machine in machines:
        rprint(str(machine))
    rprint(f"Longest queue: {longest_queue} = {estimate_time(longest_queue)}")
