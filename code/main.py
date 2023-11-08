from subprocess import CalledProcessError

import mpilammpsrun as mpilr
import shapes as s
from template import LAMMPS_EXECUTABLE, replace_templates, get_template
import nanoparticle
import poorly_coded_parser as parser


def test():
	"""
	Tests the functions in this file
	"""
	assert s.Cylinder(10, 45, 'x', (0, 0)).get_region("ns") == "region ns cylinder x 0 0 10 -22.5 22.5 units box", "Cylinder region is not correct"
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
	lammps_run.dumps[0].plot()


def plot_region(region: str):
	plot_output(replace_templates(get_template(), {
		"region": region,
		"run_steps": str(0)
	}))


def main():
	ignore = [
		# "X-Jannus_Cylinder",
		# "Y-Jannus_Cylinder",
		# "Mix05_PPP-CornerJanus_Cylinder",
		# "Mix10_PPP-CornerJanus_Cylinder",
	]
	# test()
	ok_particles = []
	not_ok_particles = []
	nanoparticles = parser.load_shapes("../Shapes/Cylinder", ignore)
	for key, nanoparticle in nanoparticles.items():
		print(f"\033[32m{key}\033[0m")
		if not any([section in key for section in ignore]):
			try:
				nanoparticle.execute(test_run=True)
				ok_particles.append(key)
			except CalledProcessError:
				print("\033[31mFailed to execute\033[0m")
				not_ok_particles.append(key)
			# nanoparticle.plot()

	for key in ok_particles:
		print(f"\033[32m{key}\033[0m")

	for key in not_ok_particles:
		print(f"\033[31m{key}\033[0m")


if __name__ == "__main__":
	main()
