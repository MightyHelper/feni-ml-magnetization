from config import LAMMPS_TEMPLATE_PATH, SLURM_TEMPLATE_PATH


class TemplateUtils:
	@staticmethod
	def replace_template(base: str, name: str, value: str) -> str:
		return base.replace(f"{{{{{name}}}}}", value)

	@staticmethod
	def replace_templates(base: str, replacements: dict) -> str:
		for key, value in replacements.items():
			base = TemplateUtils.replace_template(base, key, value)
		return base

	@staticmethod
	def get_lammps_template():
		template = open(LAMMPS_TEMPLATE_PATH, "r")
		return template.read()

	@staticmethod
	def get_slurm_template():
		template = open(SLURM_TEMPLATE_PATH, "r")
		return template.read()
