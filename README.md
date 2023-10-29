# Lammps template region evaluator

Run a template lammps simulation using a particular region of atoms.

## Use

Change LAMMPS_EXECUTABLE in main.py.

Execute `python3 main.py` from `/code`.

In the basic example, it will:
1. generate a sphere with radius 15
1. calculate it's volume algebraicly
1. create a cylinder of radius 10 with the same volume
1. create a cylinder of radius 15 with the same volume
1. print out the algebraic volumes of all the shapes
1. run lammps simulation for each of them for 0 steps
1. print out the final atom count lammps used
1. plot each region

