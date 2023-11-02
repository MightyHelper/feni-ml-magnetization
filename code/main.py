import mpilammpsrun as mpilr
import shapes as s
from template import LAMMPS_EXECUTABLE, replace_templates, get_template
import nanoparticle


def test():
	"""
	Tests the functions in this file
	"""
	assert s.Cylinder(10, 45, 'x', (0, 0, 0)).get_region() == "region ns cylinder x 0 0 10 -22.5 22.5 units box", "Cylinder region is not correct"
	assert s.Sphere(10, (0, 0, 0)).get_region() == "region ns sphere 0 0 0 10 units box", "Sphere region is not correct"


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
	sphere0 = s.Sphere(15, (0, 0, 0))
	base_volume = sphere0.get_volume()
	cyl1 = s.Cylinder.from_volume_and_radius(base_volume, 10, 'x', (0, 0, 0))
	cyl2 = s.Cylinder.from_volume_and_radius(base_volume, 15, 'x', (0, 0, 0))
	print(f"{sphere0} {base_volume=}\n{cyl1} {cyl1.get_volume()=}\n{cyl2} {cyl2.get_volume()=}")
	cyl_np = nanoparticle.Nanoparticle(cyl1)
	cyl_np.execute(test_run=False)
	print(f"Number of atoms: {cyl_np.run.dumps[0]['number_of_atoms']}")
	print(f"Magnetism: {cyl_np.magnetism}")


if __name__ == "__main__":
	main()
