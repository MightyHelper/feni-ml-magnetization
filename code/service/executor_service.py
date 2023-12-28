import logging

import execution_queue
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
    queue: execution_queue.ExecutionQueue = get_executor(at)
    for path, nanoparticle in nanoparticles:
        nanoparticle.schedule_execution(execution_queue=queue, test_run=test)
    tasks: list[SimulationTask] = queue.run()
    out_nanos: list[tuple[str, Nanoparticle]] = [(task.nanoparticle.path, task.nanoparticle) for task in tasks]
    return out_nanos
