LAMMPS_EXECUTABLE = "/home/federico/sistemas_complejos/lammps/lammps/build6/lmp"
# LAMMPS_EXECUTABLE = "python.exe -c \"import\ttime;time.sleep(100)\""
TEMPLATE_PATH = "../lammps.template"


def replace_template(base: str, name: str, value: str) -> str:
	return base.replace(f"{{{{{name}}}}}", value)


def replace_templates(base: str, replacements: dict) -> str:
	for key, value in replacements.items():
		base = replace_template(base, key, value)
	return base


def get_template():
	template = open(TEMPLATE_PATH, "r")
	return template.read()
