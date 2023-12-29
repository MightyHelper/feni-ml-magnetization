import logging
import typing

import shapes
from nanoparticle import Nanoparticle

SEED_LOCATOR = "zeed:"


class NanoparticleBuilder:
    """
    Assembles a nanoparticle
    """
    regions: list[shapes.Shape]
    atom_manipulation: list[str]
    region_name_map: dict[str, int]
    title: str
    seed_values: list[int]

    def __init__(self, title: str = "Nanoparticle"):
        self.regions = []
        self.atom_manipulation = []
        self.region_name_map = {}
        self.title = title
        self.seed_values = []

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

    def use_random_ratio(self, selector: tuple[str, str] = ("type", "1"), target_atom_type: str = "2",
                         ratio_to_convert: str = "0.35", random_seed: str = "250") -> None:
        self.atom_manipulation.append(
            f"set {selector[0]} {selector[1]} type/ratio {target_atom_type} {ratio_to_convert} {random_seed}")

    def add_named_shape(self, shape: shapes.Shape, region_name: str, extra: list[str]) -> None:
        region_id = self.register_region_name(region_name)
        self.regions.append(shape)
        self.atom_manipulation.append(shape.get_region(f"reg{region_id}") + " " + " ".join(extra))

    def register_region_name(self, region_name: str) -> int:
        self.region_name_map[region_name] = len(self.region_name_map.keys())
        logging.debug(f"Registered region {region_name} as {self.region_name_map[region_name]}")
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

    def add_set_type_subset_region(self, atom_type: str, region_name: typing.Union[str, int], count: str,
                                   seed: str) -> str:
        if isinstance(region_name, str):
            region_name = self.get_region_id_by_name(region_name)
        region_name = int(region_name)
        command = f"set region reg{region_name} type/subset {atom_type} {count} {self.new_seed(seed)}"
        self.atom_manipulation.append(command)
        return command

    def add_set_type_group(self, atom_type: str, group_name: str) -> str:
        command = f"set group {group_name} type {atom_type}"
        self.atom_manipulation.append(command)
        return command

    def add_set_type_subset_group(self, atom_type: str, group_name: str, count: str, seed: str) -> str:

        command = f"set group {group_name} type/subset {atom_type} {count} {self.new_seed(seed)}"
        self.atom_manipulation.append(command)
        return command

    def add_set_type_ratio_group(self, atom_type: str, group_name: str, ratio: str, seed: str) -> str:
        command = f"set group {group_name} type/ratio {atom_type} {ratio} {self.new_seed(seed)}"
        self.atom_manipulation.append(command)
        return command

    def add_group_type(self, atom_type: str, group_name: str):
        command = f"group {group_name} type {atom_type}"
        self.atom_manipulation.append(command)
        return command

    def add_intersect(self, regions: list[str], region_name: str, extra: list[str]):
        region_indices = [self.get_region_id_by_name(region) for region in regions]
        region_indices = [f"reg{i}" for i in region_indices]
        new_region_id = self.register_region_name(region_name)
        logging.debug(f"Intersecting regions {region_indices} into {new_region_id}")
        logging.debug(str(self.region_name_map))
        command = f"region reg{new_region_id} intersect {len(regions)} {' '.join(region_indices)} {' '.join(extra)}"
        self.atom_manipulation.append(command)
        return command

    def get_seed_count(self) -> int:
        return len(self.seed_values)

    def is_random(self) -> bool:
        return self.get_seed_count() > 0

    def build(self, seeds=None, **kwargs):
        seeds = self.seed_values if seeds is None else seeds
        nano = Nanoparticle({'title': self.title, **kwargs})
        nano.regions = self.regions
        nano.atom_manipulation = self.atom_manipulation
        nano.region_name_map = self.region_name_map
        seed_count = self.get_seed_count()
        nano.atom_manipulation = self.replace_seeds(nano.atom_manipulation, seed_count, seeds)
        logging.debug(nano.atom_manipulation)
        return nano

    def replace_seeds(self, atom_manipulation, seed_count, seeds):
        if len(seeds) != seed_count:
            raise Exception(f"Expected {seed_count} seeds, got {len(seeds)}")
        for i in range(0, seed_count):
            atom_manipulation = [x.replace(SEED_LOCATOR + str(i), str(seeds[i])) for x in atom_manipulation]
        return atom_manipulation

    def new_seed(self, seed):
        self.seed_values.append(seed)
        return SEED_LOCATOR + str(len(self.seed_values) - 1)

    def add_delete_atoms_region(self, region_name: str, keywords: str):
        # delete_atoms region v compress yes
        region_id = self.get_region_id_by_name(region_name)
        command = f"delete_atoms region reg{region_id} {keywords}"
        self.atom_manipulation.append(command)
        return command

    def add_set_type_ratio_region(self, value, region_name, ratio, seed):
        region_id = self.get_region_id_by_name(region_name)
        command = f"set region reg{region_id} type/ratio {value} {ratio} {self.new_seed(seed)}"
        self.atom_manipulation.append(command)
        return command

    def add_group_region(self, region_name, group_name):
        region_id = self.get_region_id_by_name(region_name)
        command = f"group {group_name} region reg{region_id}"
        self.atom_manipulation.append(command)
        return command

    def configure_lattice(self, lattice_type, lattice_spacing, lattice_extra):
        extra: str = " ".join(lattice_extra)
        extra = " " + extra if len(extra) > 0 else ""
        command: str = f"lattice {lattice_type} {lattice_spacing}{extra}"
        self.atom_manipulation.append(command)

    def __str__(self):
        return f"NanoparticleBuilder({self.title}, regions:{len(self.regions)}, manipulations:{len(self.atom_manipulation)}, seeds:{len(self.seed_values)})"
