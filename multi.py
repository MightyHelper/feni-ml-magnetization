# This will run in a remote cluster and will execute the nanoparticle simulations using a single job.

import os
import sys

world_size = os.getenv("OMPI_COMM_WORLD_SIZE")
world_rank = os.getenv("OMPI_COMM_WORLD_RANK")
world_local_size = os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE")
world_local_rank = os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
universe_size = os.getenv("OMPI_UNIVERSE_SIZE")
node_rank = os.getenv("OMPI_COMM_WORLD_NODE_RANK")
argv = sys.argv
all_executions = argv[1:]
my_executions = all_executions[int(world_rank)::int(world_size)]

print("Hello Nya!")
print(f"world_size = {world_size}")
print(f"world_rank = {world_rank}")
print(f"world_local_size = {world_local_size}")
print(f"world_local_rank = {world_local_rank}")
print(f"universe_size = {universe_size}")
print(f"node_rank = {node_rank}")
print(f"argv = {argv}")
print(f"all_executions = {all_executions}")
print(f"my_executions = {my_executions}")

for execution in my_executions:
	print(f"Starting {execution}")
	os.system(f"{execution}")
	print(f"Finished {execution}")
