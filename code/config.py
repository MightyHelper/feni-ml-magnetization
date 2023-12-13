import logging

# Program config
LOG_LEVEL = logging.DEBUG  # Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
LAMMPS_TEMPLATE_PATH = "../lammps.template"  # Local path pointing to the lammps template
SLURM_TEMPLATE_PATH = "../slurm.template"  # Local path pointing to the slurm template

# Local
LAMMPS_EXECUTABLE = "/home/federico/sistemas_complejos/lammps/lammps/build6/lmp"  # Path to the lammps executable in local
LOCAL_EXECUTION_PATH = "../executions"  # Path in local where the simulations will be stored

# Toko
LAMMPS_TOKO_EXECUTABLE = "/home/gdossantos/Lammps_Stable_Oct2020/lammps-29Oct20/src/lmp_g++_openmpi"  # Path to the lammps executable in toko
TOKO_EXECUTION_PATH = "~/scratch/projects/magnetism/simulations/"  # Path in toko where the simulations will be stored
TOKO_PARTITION_TO_USE = "mini"
TOKO_URL = "toko.uncu.edu.ar"
TOKO_USER = "fwilliamson"
TOKO_SBATCH = "/apps/slurm/bin/sbatch"

SLURM_SH = "slurm.sh"

# Lammps config; !!! DO NOT CHANGE !!! (Not reflected in the template)
FULL_RUN_DURATION = 300000
LAMMPS_DUMP_INTERVAL = 100000
FE_ATOM = 1
NI_ATOM = 2
