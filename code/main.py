import numpy as np

import mpilammpsrun as mpilr
import shapes as s
from template import LAMMPS_EXECUTABLE, replace_templates, get_template
import nanoparticle


def test():
	"""
	Tests the functions in this file
	"""
	assert s.Cylinder(10, 45, 'x', (0, 0, 0)).get_region("ns") == "region ns cylinder x 0 0 10 -22.5 22.5 units box", "Cylinder region is not correct"
	assert s.Sphere(10, (0, 0, 0)).get_region("ns") == "region ns sphere 0 0 0 10 units box", "Sphere region is not correct"


def count_atoms_in_region(region: str) -> int:
	folder = "../executions"
	lammps_run = mpilr.MpiLammpsRun(
		replace_templates(get_template(), {
			"region": region,
			"run_steps": str(0)
		}),
		{
			"lammps_executable": LAMMPS_EXECUTABLE,
			"cwd": folder
		},
		["iron.0.dump"]
	).execute()
	return lammps_run.dumps[0]["number_of_atoms"]


def plot_output(code: str):
	folder = "../executions"
	lammps_run = mpilr.MpiLammpsRun(
		code,
		{
			"lammps_executable": LAMMPS_EXECUTABLE,
			"cwd": folder
		},
		["iron.0.dump"]
	).execute()
	lammps_run.plot(lammps_run.dumps[0])


def plot_region(region: str):
	plot_output(replace_templates(get_template(), {
		"region": region,
		"run_steps": str(0)
	}))


def main():
	test()
	particle_factory = nanoparticle.AbstractParticleFactory.by_atom_count(1243)
	# cyl_np = particle_factory.core_shell_cylinder(12, 9)
	cyl_np = particle_factory.onion_cylinder((12.5, 30), (9, 26), (6, 22), (3, 18))
	cyl_np.execute(test_run=True)
	fe_count = cyl_np.count_atoms_of_type(nanoparticle.FE_ATOM)
	ni_count = cyl_np.count_atoms_of_type(nanoparticle.NI_ATOM)
	print(f"Real number of atoms: {cyl_np.total_atoms(0)} ({fe_count} Fe, {ni_count} Ni | {ni_count / fe_count}%)")
	ni = cyl_np.regions[1].get_lattice_point_count()
	fe = cyl_np.regions[0].get_lattice_point_count()
	print(f"Calculated atoms: FE: " + str(fe - ni))
	print(f"Calculated atoms: NI: " + str(ni))
	cyl_np.plot()


if __name__ == "__main__":
	main()
