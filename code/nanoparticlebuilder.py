import typing

import shapes
from nanoparticle import Nanoparticle


class NanoparticleBuilder:
	regions: list[shapes.Shape]
	atom_manipulation: list[str]
	region_name_map: dict[str, int]
	title: str

	def __init__(self, title: str = "Nanoparticle"):
		self.regions = []
		self.atom_manipulation = []
		self.region_name_map = {}
		self.title = title

	def add_shape(self, shape: shapes.Shape, action: str = 'create', atom_type: str = "1") -> int:
		region_index = len(self.regions)
		self.regions.append(shape)
		self.atom_manipulation.append(shape.get_region(f"reg{region_index}"))
		if action == 'create':
			self.add_create_atoms(atom_type, region_index)
		elif action == 'delete':
			self.atom_manipulation.append(f"delete_atoms {atom_type} region reg{region_index}")
		elif action == 'update':
			self.atom_manipulation.append(f"set region reg{region_index} type {atom_type}")
		else:
			raise Exception(f"Unknown action: {action}")
		return region_index

	def use_random_ratio(self, selector: tuple[str, str] = ("type", "1"), target_atom_type: str = "2", ratio_to_convert: str = "0.35", random_seed: str = "250") -> None:
		self.atom_manipulation.append(f"set {selector[0]} {selector[1]} type/ratio {target_atom_type} {ratio_to_convert} {random_seed}")

	def add_named_shape(self, shape: shapes.Shape, region_name: str) -> None:
		region_id = self.register_region_name(region_name)
		self.regions.append(shape)
		self.atom_manipulation.append(shape.get_region(f"reg{region_id}"))

	def register_region_name(self, region_name: str) -> int:
		self.region_name_map[region_name] = len(self.regions)
		return self.region_name_map[region_name]

	def get_region_id_by_name(self, region_name: str) -> int:
		return self.region_name_map[region_name]

	def add_create_atoms(self, atom_type: str, region_name: typing.Union[str, int]) -> str:
		if isinstance(region_name, str):
			region_name = self.get_region_id_by_name(region_name)
		region_name = int(region_name)
		command = f"create_atoms {atom_type} region reg{region_name}"
		self.atom_manipulation.append(command)
		return command

	def add_set_type_region(self, atom_type: str, region_name: typing.Union[str, int]) -> str:
		if isinstance(region_name, str):
			region_name = self.get_region_id_by_name(region_name)
		region_name = int(region_name)
		command = f"set region reg{region_name} type {atom_type}"
		self.atom_manipulation.append(command)
		return command

	def add_set_type_subset_region(self, atom_type: str, region_name: typing.Union[str, int], count: str, seed: str) -> str:
		if isinstance(region_name, str):
			region_name = self.get_region_id_by_name(region_name)
		region_name = int(region_name)
		command = f"set region reg{region_name} type/subset {atom_type} {count} {seed}"
		self.atom_manipulation.append(command)
		return command

	def add_set_type_group(self, atom_type: str, group_name: str) -> str:
		command = f"set group {group_name} type {atom_type}"
		self.atom_manipulation.append(command)
		return command

	def add_set_type_subset_group(self, atom_type: str, group_name: str, count: str, seed: str) -> str:
		command = f"set group {group_name} type/subset {atom_type} {count} {seed}"
		self.atom_manipulation.append(command)
		return command

	def add_set_type_ratio_group(self, atom_type: str, group_name: str, ratio: str, seed: str) -> str:
		command = f"set group {group_name} type/ratio {atom_type} {ratio} {seed}"
		self.atom_manipulation.append(command)
		return command

	def add_group_type(self, atom_type: str, group_name: str):
		command = f"group {group_name} type {atom_type}"
		self.atom_manipulation.append(command)
		return command

	def add_intersect(self, regions: list[str], region_name: str):
		region_indices = [self.get_region_id_by_name(region) for region in regions]
		region_indices = [f"reg{i}" for i in region_indices]
		new_region_id = self.register_region_name(region_name)
		command = f"region reg{new_region_id} intersect {len(regions)} {' '.join(region_indices)} units box"
		self.atom_manipulation.append(command)
		return command

	def build(self, **kwargs):
		nano = Nanoparticle({'title': self.title, **kwargs})
		nano.regions = self.regions
		nano.atom_manipulation = self.atom_manipulation
		nano.region_name_map = self.region_name_map
		return nano
