# Execute nanoparticle simulations in batch, with configurable execution queue

import random
from typing import cast

import execution_queue
import nanoparticle
import nanoparticlebuilder
import poorly_coded_parser as parser
import toko_utils
from nanoparticle import Nanoparticle
from simulation_task import SimulationTask


def get_executor(at: str) -> execution_queue.ExecutionQueue:
    """
    Executor factory
    :param at:
    :return:
    """
    if at == "local":
        return execution_queue.LocalExecutionQueue()
    elif at.startswith("local:"):
        return execution_queue.ThreadedLocalExecutionQueue(int(at.split(":")[1]))
    elif at == "toko":
        return toko_utils.TokoExecutionQueue()
    elif at.startswith("toko:"):
        return toko_utils.TokoBatchedExecutionQueue(int(at.split(":")[1]))
    else:
        raise ValueError(f"Unknown queue {at}")


def execute_nanoparticles(
        nanoparticles: list[tuple[str, Nanoparticle]],
        at: str = "local",
        test: bool = False
) -> list[tuple[str, Nanoparticle]]:
    """
    Execute a bunch of nanoparticles.
    :param nanoparticles: The list of nanoparticles to execute
    :param at: The queue to use
    :param test: Whether to use test mode or not
    :return:
    """
    queue: execution_queue.ExecutionQueue = get_executor(at)
    for path, nanoparticle in nanoparticles:
        nanoparticle.schedule_execution(execution_queue=queue, test_run=test)
    tasks: list[SimulationTask] = queue.run()
    out_nanos: list[tuple[str, Nanoparticle]] = [(task.nanoparticle.path, task.nanoparticle) for task in tasks]
    return out_nanos


def execute_single_nanoparticle(
        nanoparticle: tuple[str, Nanoparticle],
        at: str = "local",
        test: bool = False
) -> tuple[str, Nanoparticle]:
    """
    Execute only a single nanoparticle
    :param nanoparticle:
    :param at:
    :param test:
    :return:
    """
    result: list[tuple[str, Nanoparticle]] = execute_nanoparticles([nanoparticle], at=at, test=test)
    if len(result) == 0:
        raise Exception(f"Execution of nanoparticle {nanoparticle} Failed.")
    return result[0]


def build_nanoparticles_to_execute(ignore: list[str], path: str, seed: int, seed_count: int) -> list[
    tuple[str, nanoparticle.Nanoparticle]]:
    nano_builders = parser.PoorlyCodedParser.load_shapes(path, ignore)
    nanoparticles = []
    random.seed(seed)
    for key, nano in nano_builders:
        nano = cast(nanoparticlebuilder.NanoparticleBuilder, nano)
        if nano.is_random():
            for i in range(seed_count):
                seeds = [random.randint(0, 100000) for _ in range(len(nano.seed_values))]
                nanoparticles.append((key, nano.build(seeds)))
        else:
            nanoparticles.append((key, nano.build()))
    return nanoparticles
