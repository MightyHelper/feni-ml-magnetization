from config import LAMMPS_TEMPLATE_PATH, SLURM_TEMPLATE_PATH, SLURM_MULTI_TEMPLATE_PATH
from utils import read_local_file


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
		return read_local_file(LAMMPS_TEMPLATE_PATH)

	@staticmethod
	def get_slurm_template():
		return read_local_file(SLURM_TEMPLATE_PATH)

	@staticmethod
	def get_slurm_multi_template():
		return read_local_file(SLURM_MULTI_TEMPLATE_PATH)
