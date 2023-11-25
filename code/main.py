import executor as ex
import mpilammpsrun as mpilr
import shapes as s
from config import LOCAL_EXECUTION_PATH
from template import replace_templates, get_lammps_template


def test():
	"""
	Tests the functions in this file
	"""
	assert s.Cylinder(10, 45, 'x', (0, 0)).get_region("ns") == "region ns cylinder x 0 0 10 -22.5 22.5 units box", "Cylinder region is not correct"
	assert s.Sphere(10, (0, 0, 0)).get_region("ns") == "region ns sphere 0 0 0 10 units box", "Sphere region is not correct"


def count_atoms_in_region(region: str) -> int:
	lammps_run = mpilr.MpiLammpsRun(
		replace_templates(get_lammps_template(), {
			"region": region,
			"run_steps": str(0)
		}),
		{
			"cwd": LOCAL_EXECUTION_PATH
		},
		["iron.0.dump"]
	).execute()
	return lammps_run.dumps[0]["number_of_atoms"]


def plot_output(code: str):
	lammps_run = mpilr.MpiLammpsRun(
		code,
		{
			"cwd": LOCAL_EXECUTION_PATH
		},
		["iron.0.dump"]
	).execute()
	lammps_run.dumps[0].plot()


def plot_region(region: str):
	plot_output(replace_templates(get_lammps_template(), {
		"region": region,
		"run_steps": str(0)
	}))


def main():
	ex.execute_all_nanoparticles_in("../Shapes/Cylinder")


if __name__ == "__main__":
	main()
