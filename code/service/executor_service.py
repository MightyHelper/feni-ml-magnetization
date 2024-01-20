# Execute nanoparticle simulations in batch, with configurable execution queue

import random
from pathlib import Path
from typing import cast

import execution_queue
import nanoparticle
import nanoparticlebuilder
import poorly_coded_parser as parser
from config import MACHINES
from nanoparticle import Nanoparticle
from remote import toko_machine
from simulation_task import SimulationTask


def get_executor(at: str) -> execution_queue.ExecutionQueue:
    """
    Executor factory function. Returns an instance of ExecutionQueue based on the provided parameter.

    :param at: A string that determines the type of ExecutionQueue to return.
    :return: An instance of ExecutionQueue.
    """
    if at == "local":
        return execution_queue.LocalExecutionQueue(MACHINES()['local'])
    elif at.startswith("local:"):
        return execution_queue.ThreadedLocalExecutionQueue(MACHINES()['local'], int(at.split(":")[1]))
    elif at == "toko":
        return toko_machine.SlurmExecutionQueue(MACHINES()['mini'])
    elif at.startswith("toko:"):
        return toko_machine.SlurmBatchedExecutionQueue(MACHINES()['mini'], int(at.split(":")[1]))
    else:
        raise ValueError(f"Unknown queue {at}")


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
    tasks: list[SimulationTask] = queue.run()
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