import asyncio

from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn

import config.config_base
import setup_logging
import re
import os
import psutil
from rich import print as rprint
from typing import Generator, Any
from pathlib import Path
from lammps.nanoparticle import Nanoparticle
from model.live_execution import LiveExecution
from remote.machine.local_machine import LocalMachine
from remote.machine.slurm_machine import SLURMMachine

setup_logging.setup_logging()

local: LocalMachine = config.config.MACHINES()['local']


def get_batch_status() -> Generator[dict[str, Any], None, None]:
    for process in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
        try:
            if process.info['name'] == 'mpirun' and 'batch_info.txt' in process.info['cmdline']:
                yield {
                    'batch_path': Path(process.cwd()),
                    'children': list(get_batch_children(process))
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass


def get_batch_children(process: psutil.Process) -> Generator[dict[str, Any], None, None]:
    for c in process.children():
        for coc in c.children():
            for cococ in coc.children():
                yield {
                    'pid': cococ.pid,
                    'cwd': Path(cococ.cwd()),
                }


def load_batch_info(job_data: dict[str, Any]) -> Generator[LiveExecution, None, None]:
    steps = 0
    batch_info_path = job_data['batch_path'] / config.config.BATCH_INFO
    with open(batch_info_path, 'r') as file:
        for line in file.readlines():
            # id: local_path # sh -c 'cd remote_path; remote_lammps -in remote_nano_in > remote_lammps'
            match = re.match(r'(\d+): (.+) # sh -c \'cd (.+?); (.+?)\'', line)
            if match:
                pth = Path(match.group(2))
                try:
                    step = Nanoparticle.from_executed(pth).run.get_current_step()
                    steps += step
                except Exception:  # No log.lammps file yet
                    step = -1
                yield LiveExecution(
                    title=pth.name,
                    step=step,
                    folder=pth
                )

    # for child in job_data['children']:
    #     step = Nanoparticle.from_executed(child['cwd']).run.get_current_step()
    #     steps += step
    #     yield LiveExecution(
    #         title=child['cwd'].name,
    #         step=step,
    #         folder=child['cwd']
    #     )
    yield LiveExecution(
        title=f"{config.config.BATCH_EXECUTION} ({len(job_data['children'])})",
        step=steps,
        folder=job_data['batch_path']
    )


def main():
    # Example usage:
    for batch in get_batch_status():
        info = load_batch_info(batch)
        rprint("Running batch:", batch)
        for item in info:
            rprint(item)


# main()


async def main2():
    toko: SLURMMachine = config.config.MACHINES()['toko/XL']
    await toko.connect(False)
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        expand=True,
        speed_estimate_period=600,
    ) as p:
        base_remaining_time, simulations = await get_remaining_time(toko)
        remaining_time = base_remaining_time
        total_simulations = sum(simulations.values())
        task_id = p.add_task("Running simulations", total=total_simulations*300000)
        timer_task = p.add_task("Time remaining", total=base_remaining_time)
        while remaining_time > 0:
            remaining_time, simulations = await get_remaining_time(toko)
            simulations[0] += simulations[-1]
            del simulations[-1]
            result: int = sum([k * v for k, v in simulations.items()])
            p.update(task_id, completedn_f=result)
            p.update(timer_task, completed=base_remaining_time-remaining_time)
            p.refresh()
            await asyncio.sleep(5)
    await toko.disconnect()


async def get_remaining_time(toko):
    steps = [0, 100000, 200000, 300000]
    tests = [*[f"iron.{i}.dump" for i in steps], "nanoparticle.in"]
    result = await asyncio.gather(*[toko.run_cmd(f"find ~/scratch/projects/magnetism/simulations -name {i} | wc -l") for i in tests])
    simulations: dict[int, int] = {}
    simulations[-1] = int(result[-1].stdout.strip())
    for item, step in zip(result[:4], steps):
        simulations[step] = int(item.stdout.strip())
    simulations[-1] -= simulations[0]
    simulations[0] -= simulations[100000]
    simulations[100000] -= simulations[200000]
    simulations[200000] -= simulations[300000]
    simulation_time = toko.single_core_completion_time
    remaining_time = 0
    remaining_time += simulations[-1] * simulation_time * (3 / 3)
    remaining_time += simulations[000000] * simulation_time * (3 / 3)
    remaining_time += simulations[100000] * simulation_time * (2 / 3)
    remaining_time += simulations[200000] * simulation_time * (1 / 3)
    remaining_time += simulations[300000] * simulation_time * (0 / 3)
    remaining_time /= 128
    return remaining_time, simulations


asyncio.run(main2())
