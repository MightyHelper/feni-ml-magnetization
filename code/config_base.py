import logging
from pathlib import Path, PurePath, PurePosixPath


def load_machines():
    from remote.local_machine import LocalMachine
    from remote.machine_factory import MachineFactory
    return {
        'local': LocalMachine(Path(LOCAL_EXECUTION_PATH), Path(LAMMPS_EXECUTABLE)),
        'mini': MachineFactory.toko('mini'),
    }


LOG_LEVEL = logging.WARNING  # Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
LAMMPS_TEMPLATE_PATH = Path("../lammps.template")  # Local path pointing to the lammps template
SLURM_TEMPLATE_PATH = Path("../slurm.template")  # Local path pointing to the slurm template
SLURM_MULTI_TEMPLATE_PATH = Path("../slurm-multi.template")  # Local path pointing to the slurm template
SSH_MULTI_TEMPLATE_PATH = Path("../ssh-multi.template")  # Local path pointing to the slurm template
LAMMPS_EXECUTABLE = Path(
    "/home/federico/sistemas_complejos/lammps/lammps/build6/lmp")  # Path to the lammps executable in local
LOCAL_EXECUTION_PATH = Path("../executions")  # Path in local where the simulations will be stored
LOCAL_MULTI_PY = Path("../multi.py")  # Path in local to the multi.py file
LOCAL_LAMMPS_NAME_WINDOWS = "lmp.exe"
LAMMPS_TOKO_EXECUTABLE = PurePath(
    "/scratch/fwilliamson/lammps_compile/lammps/build1/lmp")  # Path to the lammps executable in toko
TOKO_EXECUTION_PATH = PurePath(
    "~/scratch/projects/magnetism/simulations/")  # Path in toko where the simulations will be stored
TOKO_PARTITION_TO_USE = "mini"
TOKO_URL = "toko.uncu.edu.ar"
TOKO_USER = "fwilliamson"
TOKO_SBATCH = PurePosixPath("/apps/slurm/bin/sbatch")
TOKO_SQUEUE = PurePosixPath("/apps/slurm/bin/squeue")
TOKO_SCONTROL = PurePosixPath("/apps/slurm/bin/scontrol")
TOKO_SINFO = PurePosixPath("/apps/slurm/bin/sinfo")
TOKO_COPY_SCRIPT = "rsync"  # rsync or scp
BATCH_INFO_PATH = PurePath("batch_info.txt")
SLURM_SH = PurePath("slurm.sh")
RUN_SH = PurePath("run.sh")
MACHINES = load_machines
FULL_RUN_DURATION = 300000
LAMMPS_DUMP_INTERVAL = 100000
FE_ATOM = 1
NI_ATOM = 2
BATCH_EXECUTION = "Batch execution"
FINISHED_JOB = "Finished job"
NANOPARTICLE_IN = PurePath("nanoparticle.in")
DESIRED_ATOM_COUNT = 1250
DESIRED_NI_RATIO = 0.3
DESIRED_FE_RATIO = 1 - DESIRED_NI_RATIO
DESIRED_TOLERANCE = 0.1  # 10% of the actual value
DESIRED_MAX_RATIO_VARIANCE = DESIRED_TOLERANCE * DESIRED_NI_RATIO
DESIRED_NI_ATOM_COUNT = DESIRED_NI_RATIO * DESIRED_ATOM_COUNT
DESIRED_FE_ATOM_COUNT = DESIRED_FE_RATIO * DESIRED_ATOM_COUNT
DESIRED_MAX_ATOM_COUNT_VARIANCE = DESIRED_TOLERANCE * DESIRED_ATOM_COUNT
