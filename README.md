# Lammps template region evaluator

Run a template lammps simulation using a particular region of atoms.

## Use

Change LAMMPS_EXECUTABLE in [template.py](code/template.py).

Execute `python3 main.py` from `/code`.

In the basic example, it will:

1. Generate a sphere with radius 15
2. Calculate the volume of the sphere
3. Create a core-shell cylinder with an inner and outer radius, with the appropriate height to have the same atom count as requested
4. It will also count the number of atoms in the core and shell
5. It will then plot the atoms.

NOTE: The lattice is causing the atoms to be placed in a grid, so the cylinders are not perfect and the atom ratios are
not exact.
