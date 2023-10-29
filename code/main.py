import mpilammpsrun as mpilr
import shapes as s


def replace_template(name: str, value: str, base: str) -> str:
	return base.replace(f"{{{{{name}}}}}", value)


def replace_templates(replacements: dict, base: str) -> str:
	for key, value in replacements.items():
		base = replace_template(key, value, base)
	return base


def test():
	"""
	Tests the functions in this file
	"""
	assert s.Cylinder(10, 45, 'x', (0, 0, 0)).get_region() == "region ns cylinder x 0 0 10 -22.5 22.5 units box", "Cylinder region is not correct"
	assert s.Sphere(10, (0, 0, 0)).get_region() == "region ns sphere 0 0 0 10 units box", "Sphere region is not correct"


def count_atoms_in_region(region: str) -> int:
	folder = "../executions"
	lammps_run = mpilr.MpiLammpsRun(
		replace_templates({
			"region": region,
			"run_steps": str(0)
		}, get_template()),
		{
			"lammps_executable": "/home/federico/sistemas_complejos/lammps/lammps/build6/lmp",
			"cwd": folder
		},
		[folder + "/iron.0.dump"]
	)
	return lammps_run.dumps[0]["number_of_atoms"]


def get_template():
	template = open("../lammps.template", "r")
	return template.read()


def main():
	test()
	sphere0 = s.Sphere(15, (0, 0, 0))
	base_volume = sphere0.get_volume()
	cyl1 = s.Cylinder.from_volume_and_radius(base_volume, 10, 'x', (0, 0, 0))
	cyl2 = s.Cylinder.from_volume_and_radius(base_volume, 15, 'x', (0, 0, 0))
	print(f"{sphere0} {base_volume=}\n{cyl1} {cyl1.get_volume()=}\n{cyl2} {cyl2.get_volume()=}")
	atoms_cyl1 = count_atoms_in_region(cyl1.get_region())
	atoms_cyl2 = count_atoms_in_region(cyl2.get_region())
	atoms_sph = count_atoms_in_region(sphere0.get_region())
	print(f"{atoms_cyl1=} {atoms_cyl2=} {atoms_sph=}")


if __name__ == "__main__":
	main()
