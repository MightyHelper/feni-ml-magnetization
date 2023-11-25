LAMMPS_EXECUTABLE = "/home/federico/sistemas_complejos/lammps/lammps/build6/lmp"
LAMMPS_TOKO_EXECUTABLE = "/home/gdossantos/Lammps_Stable_Oct2020/lammps-29Oct20/src/lmp_g++_openmpi"
# LAMMPS_EXECUTABLE = "C:\\Users\\feder\\AppData\\Local\\Programs\\Python\\Python310\\python.exe -c \"import\ttime;time.sleep(100)\""
LAMMPS_TEMPLATE_PATH = "../lammps.template"
SLURM_TEMPLATE_PATH = "../slurm.template"


def replace_template(base: str, name: str, value: str) -> str:
	return base.replace(f"{{{{{name}}}}}", value)


def replace_templates(base: str, replacements: dict) -> str:
	for key, value in replacements.items():
		base = replace_template(base, key, value)
	return base


def get_lammps_template():
	template = open(LAMMPS_TEMPLATE_PATH, "r")
	return template.read()


def get_slurm_template():
	template = open(SLURM_TEMPLATE_PATH, "r")
	return template.read()
