from pathlib import Path

import typer
from rich import print as rprint

from lammps.nanoparticle import Nanoparticle
from remote.execution_queue.execution_queue import ExecutionQueue
from remote.execution_queue.mixed_execution_queue import MixedExecutionQueue
from remote.execution_queue.slurm_execution_queue import estimate_slurm_time, estimate_minutes, minutes_to_slurm
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
        np.schedule_execution(execution_queue=queue, test_run=True)
    if isinstance(queue, MixedExecutionQueue):
        queue.schedule()
    estimated_min = max(_render_queue(queue))
    rprint(f"Estimated time: {minutes_to_slurm(estimated_min)}")

def _render_queue(queue: ExecutionQueue) -> list[float]:
    if isinstance(queue, MixedExecutionQueue):
        return [qmin for q in queue.queues for qmin in _render_queue(q)]
    else:
        qlen = len(queue.queue)
        qcores = queue.parallelism_count
        qperf = queue.remote.single_core_performance
        qmin = estimate_minutes(qlen, qcores, qperf)
        qtime = estimate_slurm_time(qlen, qcores, qperf)
        rprint(f"{queue.remote.name:12} ({queue.parallelism_count:3} cores): {len(queue.queue):4} tasks = {qtime}")
        return [qmin]
