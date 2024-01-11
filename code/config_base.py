import logging
import multiprocessing

from model.machine import Machine, SLURMMachine, get_toko_cores

LOG_LEVEL = logging.WARNING  # Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
LAMMPS_TEMPLATE_PATH = "../lammps.template"  # Local path pointing to the lammps template
SLURM_TEMPLATE_PATH = "../slurm.template"  # Local path pointing to the slurm template
SLURM_MULTI_TEMPLATE_PATH = "../slurm-multi.template"  # Local path pointing to the slurm template
LAMMPS_EXECUTABLE = "/home/federico/sistemas_complejos/lammps/lammps/build6/lmp"  # Path to the lammps executable in local
LOCAL_EXECUTION_PATH = "../executions"  # Path in local where the simulations will be stored
LOCAL_MULTI_PY = "../multi.py"  # Path in local to the multi.py file
LAMMPS_TOKO_EXECUTABLE = "/scratch/fwilliamson/lammps_compile/lammps/build1/lmp"  # Path to the lammps executable in toko
TOKO_EXECUTION_PATH = "~/scratch/projects/magnetism/simulations/"  # Path in toko where the simulations will be stored
TOKO_PARTITION_TO_USE = "mini"
TOKO_URL = "toko.uncu.edu.ar"
TOKO_USER = "fwilliamson"
TOKO_SBATCH = "/apps/slurm/bin/sbatch"
TOKO_SQUEUE = "/apps/slurm/bin/squeue"
TOKO_SCONTROL = "/apps/slurm/bin/scontrol"
TOKO_COPY_SCRIPT = "rsync"  # rsync or scp
TOKO_BATCH_INFO_PATH = "batch_info.txt"
SLURM_SH = "slurm.sh"
MACHINES: list[Machine] = [
    SLURMMachine(
        name="toko",
        hostname=TOKO_URL,
        cores=get_toko_cores(TOKO_PARTITION_TO_USE),
        user=TOKO_USER,
        task_queue=[],
        partition=TOKO_PARTITION_TO_USE,
        node_id=1,
    ), Machine(
        name="local",
        cores=multiprocessing.cpu_count,
        task_queue=[],
    )
]
FULL_RUN_DURATION = 300000
LAMMPS_DUMP_INTERVAL = 100000
FE_ATOM = 1
NI_ATOM = 2
BATCH_EXECUTION = "Batch execution"
NANOPARTICLE_IN = "nanoparticle.in"
DESIRED_ATOM_COUNT = 1250
DESIRED_NI_RATIO = 0.3
DESIRED_FE_RATIO = 1 - DESIRED_NI_RATIO
DESIRED_TOLERANCE = 0.1  # 10% of the actual value
DESIRED_MAX_RATIO_VARIANCE = DESIRED_TOLERANCE * DESIRED_NI_RATIO
DESIRED_NI_ATOM_COUNT = DESIRED_NI_RATIO * DESIRED_ATOM_COUNT
DESIRED_FE_ATOM_COUNT = DESIRED_FE_RATIO * DESIRED_ATOM_COUNT
DESIRED_MAX_ATOM_COUNT_VARIANCE = DESIRED_TOLERANCE * DESIRED_ATOM_COUNT
