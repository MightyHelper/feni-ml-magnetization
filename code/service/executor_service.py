# Execute nanoparticle simulations in batch, with configurable execution queue
import logging
import random
from pathlib import Path
from typing import cast

from rich.progress import Progress, SpinnerColumn, MofNCompleteColumn, TimeElapsedColumn, TaskID

from remote.execution_queue import local_execution_queue, execution_queue, slurm_execution_queue, mixed_execution_queue
from lammps import nanoparticle, poorly_coded_parser as parser, nanoparticlebuilder
from config.config import MACHINES
from lammps.nanoparticle import Nanoparticle
from remote.execution_queue.execution_queue import ExecutionQueue
from remote.machine.local_machine import LocalMachine
from remote.machine.machine import Machine
from remote.machine.slurm_machine import SLURMMachine
from remote.machine.ssh_machine import SSHMachine, SSHBatchedExecutionQueue
from lammps.simulation_task import SimulationTask


def get_execution_queue(machine: Machine, n_threads: int, local_machine: LocalMachine):
    if isinstance(machine, LocalMachine):
        if n_threads is None:
            return local_execution_queue.LocalExecutionQueue(machine)
        return local_execution_queue.ThreadedLocalExecutionQueue(machine, n_threads)
    elif isinstance(machine, SLURMMachine):
        if n_threads is None:
            # return slurm_execution_queue.SlurmExecutionQueue(machine)
            n_threads = 1
        return slurm_execution_queue.SlurmBatchedExecutionQueue(machine, local_machine, n_threads)
    elif isinstance(machine, SSHMachine):
        if n_threads is None:
            n_threads = 1
        return SSHBatchedExecutionQueue(machine, local_machine, n_threads)


def get_executor(at: str) -> execution_queue.ExecutionQueue:
    """
    Executor factory function. Returns an instance of ExecutionQueue based on the provided parameter.

    :param at: A string that determines the type of ExecutionQueue to return.
    :return: An instance of ExecutionQueue.
    """
    if at == "all":
        machines = MACHINES()
        queues = [get_executor(f"{machine.name}:{machine.cores}") for machine in machines.values() if machine.name != "local-ssh"]
        return mixed_execution_queue.MixedExecutionQueue(queues)
    if "," in at:
        ats = at.split(",")
        queues = [get_executor(a) for a in ats]
        return mixed_execution_queue.MixedExecutionQueue(queues)
    machine_name, *threads = at.split(":")
    n_threads: int | None = int(threads[0]) if len(threads) > 0 else None
    machines = MACHINES()
    for name, machine in machines.items():
        if machine_name == name:
            return get_execution_queue(machine, n_threads, machines["local"])
    raise ValueError(f"Unknown queue {at} (known queues: {list(machines.keys())})")


def _handle_update(prog: Progress, task_id: TaskID):
    def inner(progress: int, total: int, task: tuple[SimulationTask, str | None], sender: ExecutionQueue):
        prog.update(task_id, completed=progress, total=total, task=task[0].nanoparticle.local_path)

    return inner


def execute_nanoparticles(
    nanoparticles: list[tuple[str, Nanoparticle]],
    at: str = "local",
    test: bool = False
) -> list[tuple[str, Nanoparticle]]:
    """
    Executes a list of nanoparticles using the specified execution queue.

    :param nanoparticles: A list of tuples, each containing a string and a Nanoparticle object.
    :param at: A string that determines the type of ExecutionQueue to use.
    :param test: A boolean that determines whether to use test mode or not.
    :return: A list of tuples, each containing a string and a Nanoparticle object.
    """
    queue: execution_queue.ExecutionQueue = get_executor(at)
    for path, np in nanoparticles:
        np.schedule_execution(execution_queue=queue, test_run=test)
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        expand=True,
        refresh_per_second=20
    ) as prog:
        task_id = prog.add_task("Executing", total=len(nanoparticles))
        queue.listen(ExecutionQueue.PROGRESS, _handle_update(prog, task_id))
        tasks: list[SimulationTask] = queue.run()
        prog.remove_task(task_id)
    out_nanos: list[tuple[str, Nanoparticle]] = [(task.nanoparticle.local_path, task.nanoparticle) for task in tasks]
    return out_nanos


def execute_single_nanoparticle(
    np: tuple[str, Nanoparticle],
    at: str = "local",
    test: bool = False
) -> tuple[str, Nanoparticle]:
    """
    Executes a single nanoparticle using the specified execution queue.

    :param np: A tuple containing a string and a Nanoparticle object.
    :param at: A string that determines the type of ExecutionQueue to use.
    :param test: A boolean that determines whether to use test mode or not.
    :return: A tuple containing a string and a Nanoparticle object.
    """
    result: list[tuple[str, Nanoparticle]] = execute_nanoparticles([np], at=at, test=test)
    if len(result) == 0:
        raise Exception(f"Execution of nanoparticle {np} Failed.")
    return result[0]


def add_extra_nanoparticles(
    nano_builders: list[tuple[str, nanoparticlebuilder.NanoparticleBuilder]],
    seed: int,
    seed_count: int
) -> list[tuple[str, nanoparticle.Nanoparticle]]:
    """
    Adds extra nanoparticles with different seed values to the list of nanoparticles.

    :param nano_builders: A list of tuples, each containing a string and a NanoparticleBuilder object.
    :param seed: An integer used to seed the random number generator.
    :param seed_count: An integer that determines the number of extra nanoparticles to add.
    :return: A list of tuples, each containing a string and a Nanoparticle object.
    """
    random.seed(seed)
    nanoparticles = []
    for key, nano in nano_builders:
        nano = cast(nanoparticlebuilder.NanoparticleBuilder, nano)
        if not nano.is_random():
            nanoparticles.append((key, nano.build()))
            continue
        for i in range(seed_count):
            seeds = [random.randint(0, 100000) for _ in range(len(nano.seed_values))]
            nanoparticles.append((key, nano.build(seeds)))
    return nanoparticles


def build_nanoparticles_to_execute(ignore: list[str], path: Path, seed: int, seed_count: int) -> list[
    tuple[str, nanoparticle.Nanoparticle]]:
    """
    Builds a list of nanoparticles to execute.

    :param ignore: A list of strings that determines which shapes to ignore.
    :param path: A Path object that specifies the location of the shapes.
    :param seed: An integer used to seed the random number generator.
    :param seed_count: An integer that determines the number of extra nanoparticles to add.
    :return: A list of tuples, each containing a string and a Nanoparticle object.
    """
    nano_builders = parser.PoorlyCodedParser.load_shapes(path, ignore)
    return add_extra_nanoparticles(list(nano_builders), seed, seed_count)
