# This will run in a remote cluster and will execute the nanoparticle simulations using a single job.

import os
import sys
import time
import subprocess

print("Hello Nya 🐾!", flush=True)  # This shall stay in the final release
world_size = os.getenv("OMPI_COMM_WORLD_SIZE") or os.getenv("MPI_LOCALNRANKS")
world_rank = os.getenv("OMPI_COMM_WORLD_RANK") or os.getenv("MPI_LOCALRANKID")
world_local_size = os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE") or os.getenv("MPI_LOCALNRANKS")
world_local_rank = os.getenv("OMPI_COMM_WORLD_LOCAL_RANK") or os.getenv("MPI_LOCALRANKID")
universe_size = os.getenv("OMPI_UNIVERSE_SIZE") or os.getenv("MPI_LOCALNRANKS")
node_rank = os.getenv("OMPI_COMM_WORLD_NODE_RANK") or os.getenv("MPI_LOCALRANKID")

print(f"world_size = {world_size}", flush=True)
print(f"world_rank = {world_rank}", flush=True)
print(f"world_local_size = {world_local_size}", flush=True)
print(f"world_local_rank = {world_local_rank}", flush=True)
print(f"universe_size = {universe_size}", flush=True)
print(f"node_rank = {node_rank}", flush=True)

with open(sys.argv[1], "r") as f:
    all_executions = f.read().split("\n")
    all_executions = [line.split(" # ")[-1] for line in all_executions if line != ""]
    my_executions = all_executions[int(world_rank)::int(world_size)]

print(f"argv = {sys.argv}", flush=True)
print(f"all_executions = {all_executions}", flush=True)
print(f"my_executions = {my_executions}", flush=True)

important_env = [
    "LD_LIBRARY_PATH",
    "PATH",
    "PYTHONPATH",
    "LIBRARY_PATH",
    "MODULESHOME",
    "MANPATH",
    "MPI_HOME",
    "MPI_SYSCONFIG",
    "LOADEDMODULES",
    "MPI_BIN",
    "MPI_MAN",
    "MPI_INCLUDE",
    "MPI_LIB",
    "MPI_LIBDIR",
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "OMPI_UNIVERSE_SIZE",
    "OMPI_COMM_WORLD_NODE_RANK",
    "MPI_LOCALNRANKS",
    "MPI_LOCALRANKID",
]
env = {k: v for k, v in os.environ.items() if k in important_env}

for execution in my_executions:
    print(f"[{world_rank} {time.time():.2f}] Starting {execution}", flush=True)

    subprocess.run(f"{execution}", env=env, shell=True)
    print(f"[{world_rank} {time.time():.2f}] Finished {execution}", flush=True)
