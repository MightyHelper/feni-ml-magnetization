import logging
from multiprocessing.pool import ThreadPool
from pathlib import Path, PurePosixPath


def load_machines():
    from remote.machine.local_machine import LocalMachine
    from remote.machine.machine_factory import MachineFactory
    return {
        'local': LocalMachine(Path(LOCAL_EXECUTION_PATH), Path(LAMMPS_EXECUTABLE)),
        'toko/mini': MachineFactory.toko('mini'),
        'toko/Small': MachineFactory.toko('Small'),
        'toko/Large': MachineFactory.toko('Large'),
        'toko/XL': MachineFactory.toko('XL'),
        'toko/XXL': MachineFactory.toko('XXL'),
        'toko/prueba': MachineFactory.toko('prueba'),
    }


LOG_LEVEL = logging.WARNING  # Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
LAMMPS_TEMPLATE_PATH = Path("../lammps.template").resolve().expanduser()  # Local path pointing to the lammps template
SLURM_TEMPLATE_PATH = Path("../slurm.template").resolve().expanduser()  # Local path pointing to the slurm template
SLURM_MULTI_TEMPLATE_PATH = Path("../slurm-multi.template").resolve().expanduser()  # Local path pointing to the slurm template
SSH_MULTI_TEMPLATE_PATH = Path("../ssh-multi.template").resolve().expanduser()  # Local path pointing to the slurm template
LAMMPS_EXECUTABLE = Path(
    "/home/federico/sistemas_complejos/lammps/lammps/build6/lmp")  # Path to the lammps executable in local
LOCAL_EXECUTION_PATH = Path("../executions").resolve().expanduser()  # Path in local where the simulations will be stored
LOCAL_MULTI_PY = Path("../multi.py").resolve().expanduser()  # Path in local to the multi.py file
LOCAL_LAMMPS_NAME_WINDOWS = "lmp.exe"
TOKO_PARTITION_TO_USE = "mini"
TOKO_URL = "toko.uncu.edu.ar"
TOKO_USER = "fwilliamson"
LAMMPS_TOKO_EXECUTABLE = PurePosixPath("/scratch/fwilliamson/lammps_compile/lammps/build1/lmp")  # Path to the lammps executable in toko
TOKO_EXECUTION_PATH = PurePosixPath("/scratch/fwilliamson/projects/magnetism/simulations/")  # Path in toko where the simulations will be stored
TOKO_SBATCH = PurePosixPath("/apps/slurm/bin/sbatch")
TOKO_SQUEUE = PurePosixPath("/apps/slurm/bin/squeue")
TOKO_SCONTROL = PurePosixPath("/apps/slurm/bin/scontrol")
TOKO_SINFO = PurePosixPath("/apps/slurm/bin/sinfo")
BATCH_INFO = "batch_info.txt"
SLURM_SH = "slurm.sh"
RUN_SH = "run.sh"
EXEC_LS_POOL_TYPE = ThreadPool
MACHINES = load_machines
FULL_RUN_DURATION = 300000
LAMMPS_DUMP_INTERVAL = 100000
FE_ATOM = 1
NI_ATOM = 2
BATCH_EXECUTION: str = "Batch execution"  # Constant
FINISHED_JOB: str = "Finished job"  # Constant
NANOPARTICLE_IN: str = "nanoparticle.in"  # Constant
LOG_LAMMPS: str = "log.lammps"  # Constant
DESIRED_ATOM_COUNT = 1250
DESIRED_NI_RATIO = 0.3
DESIRED_FE_RATIO = 1 - DESIRED_NI_RATIO
DESIRED_TOLERANCE = 0.1  # 10% of the actual value
DESIRED_MAX_RATIO_VARIANCE = DESIRED_TOLERANCE * DESIRED_NI_RATIO
DESIRED_NI_ATOM_COUNT = DESIRED_NI_RATIO * DESIRED_ATOM_COUNT
DESIRED_FE_ATOM_COUNT = DESIRED_FE_RATIO * DESIRED_ATOM_COUNT
DESIRED_MAX_ATOM_COUNT_VARIANCE = DESIRED_TOLERANCE * DESIRED_ATOM_COUNT
