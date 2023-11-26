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
TOKO_PARTITION_TO_USE = "mini"
