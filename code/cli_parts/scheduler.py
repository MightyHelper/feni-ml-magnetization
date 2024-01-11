import typer

import config
from model.machine import Machine, SSHMachine
from model.machine_factory import MachineFactory
from nanoparticle import Nanoparticle
from rich import print as rprint
from service import executor_service
from service.scheduler_service import SchedulerService
from simulation_task import SimulationTask
from toko_utils import estimate_time

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
        "../Shapes",
        seed,
        seed_count
    )
    tasks: list[SimulationTask] = [nanoparticle.get_simulation_task() for _, nanoparticle in nanoparticles]

    execution_plan: tuple[list[Machine], int] = SchedulerService.schedule(
        machines=config.MACHINES(),
        tasks=tasks
    )
    matchines, longest_queue = execution_plan
    for machine in matchines:
        rprint(str(machine))
    rprint(f"Longest queue: {longest_queue} = {estimate_time(longest_queue)}")
