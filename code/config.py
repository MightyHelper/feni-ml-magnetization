import logging

# Program config
LOG_LEVEL = logging.INFO  # Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Local
LAMMPS_EXECUTABLE = "/home/federico/sistemas_complejos/lammps/lammps/build6/lmp"
LAMMPS_TEMPLATE_PATH = "../lammps.template"
LOCAL_EXECUTION_PATH = "../executions"

# Toko
LAMMPS_TOKO_EXECUTABLE = "/home/gdossantos/Lammps_Stable_Oct2020/lammps-29Oct20/src/lmp_g++_openmpi"
SLURM_TEMPLATE_PATH = "../slurm.template"
SLURM_SH = "slurm.sh"
TOKO_PARTITION_TO_USE = "mini"
TOKO_URL = "toko.uncu.edu.ar"
USER = "fwilliamson"

# Lammps config; !!! DO NOT CHANGE !!! (Not reflected in the template)
FULL_RUN_DURATION = 300000
LAMMPS_DUMP_INTERVAL = 100000
FE_ATOM = 1
NI_ATOM = 2
