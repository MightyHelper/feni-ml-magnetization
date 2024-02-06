from pathlib import Path

import typer
from rich import print as rprint

from lammps.nanoparticle import Nanoparticle
from remote.execution_queue.execution_queue import ExecutionQueue
from remote.execution_queue.mixed_execution_queue import MixedExecutionQueue, render_queue_plan
from remote.execution_queue.slurm_execution_queue import minutes_to_slurm
from service import executor_service
from service.executor_service import get_executor

sched = typer.Typer(add_completion=False, no_args_is_help=True, name="sched")


@sched.command()
def schedule(
    at: str = typer.Option(
        "all",
        help="Where to run the simulations",
        show_default=True
    ),
    seed: int = typer.Option(
        0,
        help="Seed to start with",
        show_default=True
    ),
    seed_count: int = typer.Option(
        1,
        help="Number of seeds to run",
        show_default=True
    ),
    test: bool = typer.Option(
        False,
        help="Run a test run",
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
    queue: ExecutionQueue = get_executor(at)
    for path, np in nanoparticles:
        np.schedule_execution(execution_queue=queue, test_run=test)
    if isinstance(queue, MixedExecutionQueue):
        queue.schedule(test)
    estimated_min = max(render_queue_plan(queue, is_test=test))
    rprint(f"Estimated time: {minutes_to_slurm(estimated_min)}")
