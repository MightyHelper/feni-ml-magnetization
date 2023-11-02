# Lammps template region evaluator

Run a template lammps simulation using a particular region of atoms.

## Use

Change LAMMPS_EXECUTABLE in [template.py](code/template.py).

Execute `python3 main.py` from `/code`.

In the basic example, it will:
1. generate a sphere with radius 15
2. Run a lammps simulation for that nanoparticle
3. It will also run the pre-existing analysis on the shape
