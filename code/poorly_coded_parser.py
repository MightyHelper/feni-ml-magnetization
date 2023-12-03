import logging
import os
import re

import nanoparticlebuilder
import template

import shapes as s
from shapes import Cylinder, Sphere, Plane, Cone, Prism


def recursive_input_search(path: str):
	for file in os.listdir(path):
		if file.startswith("Test"):
			continue
		if os.path.isdir(f"{path}/{file}"):
			yield from recursive_input_search(f"{path}/{file}")
		elif file.endswith(".in"):
			yield f"{path}/{file}"


def sorted_recursive_input_search(path: str):
	return sorted([*recursive_input_search(path)])


def parse_region(line: str, nano: nanoparticlebuilder.NanoparticleBuilder) -> Cylinder | None | Sphere | Plane | Cone | Prism:
	# Remove multiple spaces
	line = split_command(line)
	region_name = line[1]
	region_type = line[2]
	region_args = line[3:]
	# Parse region
	if region_type == "cylinder":
		# region cylinder x 0 0 10 -22.5 22.5 units box
		axis = region_args[0]
		coord_a = float(region_args[1])
		coord_b = float(region_args[2])
		radius = float(region_args[3])
		neg_length = float(region_args[4])
		pos_length = float(region_args[5])
		extra = region_args[6:]
		assert extra[0] == "units", f"Unknown units: {extra[0]}"
		assert extra[1] == "box", f"Unknown box: {extra[1]}"
		extra = extra[2:]
		full_length = abs(pos_length - neg_length)
		coord_c = (pos_length + neg_length) / 2.0
		shape = None
		if axis == "x":
			shape = s.Cylinder(radius, full_length, axis, (coord_c, coord_a, coord_b), check_in_box=False)
		elif axis == "y":
			shape = s.Cylinder(radius, full_length, axis, (coord_a, coord_c, coord_b), check_in_box=False)
		elif axis == "z":
			shape = s.Cylinder(radius, full_length, axis, (coord_a, coord_b, coord_c), check_in_box=False)
		assert is_correct_parsing(line, shape.get_region(region_name)), f"Region {region_name} is not parsed correctly: \n{split_command(shape.get_region(region_name))} != \n{line}"
		nano.add_named_shape(shape, region_name, extra)
		logging.debug(shape)
		return shape
	elif region_type == "sphere":
		# region sphere 0 0 0 10 units box
		coord_a = float(region_args[0])
		coord_b = float(region_args[1])
		coord_c = float(region_args[2])
		radius = float(region_args[3])
		extra = region_args[4:]
		assert extra[0] == "units", f"Unknown units: {extra[0]}"
		assert extra[1] == "box", f"Unknown box: {extra[1]}"
		extra = extra[2:]
		shape = s.Sphere(radius, (coord_a, coord_b, coord_c))
		assert is_correct_parsing(line, shape.get_region(region_name)), f"Region {region_name} is not parsed correctly: \n{split_command(shape.get_region(region_name))} != \n{line}"
		nano.add_named_shape(shape, region_name, extra)
		logging.debug(shape)
		return shape
	elif region_type == "plane":
		# region plane 0 0 0 0 0 1 units box
		coord_a = float(region_args[0])
		coord_b = float(region_args[1])
		coord_c = float(region_args[2])
		normal_a = float(region_args[3])
		normal_b = float(region_args[4])
		normal_c = float(region_args[5])
		extra = region_args[6:]
		assert extra[0] == "units", f"Unknown units: {extra[0]}"
		assert extra[1] == "box", f"Unknown box: {extra[1]}"
		extra = extra[2:]
		shape = s.Plane((coord_a, coord_b, coord_c), (normal_a, normal_b, normal_c))
		assert is_correct_parsing(line, shape.get_region(region_name)), f"Region {region_name} is not parsed correctly: \n{split_command(shape.get_region(region_name))} != \n{line}"
		nano.add_named_shape(shape, region_name, extra)
		logging.debug(shape)
		return shape
	elif region_type == "cone":
		# region_args = ['z', '0.0', '0.0', '18', '1', '-21', '21', 'units', 'box']
		axis = region_args[0]
		coord_a = float(region_args[1])
		coord_b = float(region_args[2])
		radlo = float(region_args[3])
		radhi = float(region_args[4])
		lo = float(region_args[5])
		hi = float(region_args[6])
		extra = region_args[7:]
		assert extra[0] == "units", f"Unknown units: {extra[0]}"
		assert extra[1] == "box", f"Unknown box: {extra[1]}"
		shape = s.Cone(axis, coord_a, coord_b, radlo, radhi, lo, hi)
		assert is_correct_parsing(line, shape.get_region(region_name)), f"Region {region_name} is not parsed correctly: \n{split_command(shape.get_region(region_name))} != \n{line}"
		nano.add_named_shape(shape, region_name, extra)
		logging.debug(shape)
		return shape
	elif region_type == "prism":
		# region 		sq prism -20 20 -3 3 -16 16 0 0 0 units box
		xlo = float(region_args[0])
		xhi = float(region_args[1])
		ylo = float(region_args[2])
		yhi = float(region_args[3])
		zlo = float(region_args[4])
		zhi = float(region_args[5])
		xy = float(region_args[6])
		xz = float(region_args[7])
		yz = float(region_args[8])
		extra = region_args[9:]
		assert extra[0] == "units", f"Unknown units: {extra[0]}"
		assert extra[1] == "box", f"Unknown box: {extra[1]}"
		extra = extra[2:]
		shape = s.Prism(xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz)
		assert is_correct_parsing(line, shape.get_region(region_name)), f"Region {region_name} is not parsed correctly: \n{split_command(shape.get_region(region_name))} != \n{line}"
		nano.add_named_shape(shape, region_name, extra)
		logging.debug(shape)
		return shape
	elif region_type == "intersect":
		n = int(region_args[0])
		reg_ids = region_args[1:1 + n]
		extra = region_args[1 + n:]
		command = nano.add_intersect(reg_ids, region_name, extra)
		logging.debug(command)
		return None
	else:
		raise ValueError(f"Unknown region type: {region_type}")


def is_correct_parsing(line: list[str], command: str) -> bool:
	parsed_command = split_command(command)
	for x in range(0, len(parsed_command)):
		if (parsed_command[x] != line[x]
			and f"{parsed_command[x]}.0" != line[x]
			and f"{line[x]}.0" != parsed_command[x]
			and not parsed_command[x].startswith(nanoparticlebuilder.SEED_LOCATOR)
			and abs(float(parsed_command[x]) - float(line[x])) > 0.0001):
			logging.debug(f"[red]{parsed_command[x]} != {line[x]}[/red]")
			return False
	return True


def parse_create_atoms(line: str, nano: nanoparticlebuilder.NanoparticleBuilder) -> None:
	# Remove multiple spaces
	line = split_command(line)
	atom_type = line[1]
	selector_type = line[2]
	selector_args = line[3:]
	# Parse region
	if selector_type == "region":
		result = nano.add_create_atoms(atom_type, selector_args[0])
		logging.debug(result)
		return
	else:
		raise ValueError(f"Unknown selector type: {selector_type}")


def split_command(line: str) -> list[str]:
	return re.split("\\s+", line)


def parse_set(line: str, nano: nanoparticlebuilder.NanoparticleBuilder) -> None:
	# Remove multiple spaces
	line = split_command(line)
	set_type = line[1]
	set_args = line[2:]
	# Parse region
	if set_type == "region":
		region_name = set_args[0]
		prop = set_args[1]
		if prop == "type":
			value = set_args[2]
			result = nano.add_set_type_region(value, region_name)
			logging.debug(result)
			return
		elif prop == "type/subset":
			value = set_args[2]
			count = set_args[3]
			seed = set_args[4]
			result = nano.add_set_type_subset_region(value, region_name, count, seed)
			logging.debug(result)
			return
		elif prop == "type/ratio":
			value = set_args[2]
			ratio = set_args[3]
			seed = set_args[4]
			result = nano.add_set_type_ratio_region(value, region_name, ratio, seed)
			logging.debug(result)
			return
		else:
			raise ValueError(f"Unknown set property: {prop}")
	elif set_type == "group":
		group_name = set_args[0]
		prop = set_args[1]
		if prop == "type/subset":
			value = set_args[2]
			count = set_args[3]
			seed = set_args[4]
			result = nano.add_set_type_subset_group(value, group_name, count, seed)
			assert is_correct_parsing(line, result), f"Set type subset group {group_name} is not parsed correctly: \n{result} != \n{line}"
			logging.debug(result)
			return
		elif prop == "type/ratio":
			value = set_args[2]
			ratio = set_args[3]
			seed = set_args[4]
			result = nano.add_set_type_ratio_group(value, group_name, ratio, seed)
			assert is_correct_parsing(line, result), f"Set type ratio group {group_name} is not parsed correctly: \n{result} != \n{line}"
			logging.debug(result)
			return
		else:
			raise ValueError(f"Unknown set property: {prop}")
	else:
		raise ValueError(f"Unknown selector type: {set_type}")


def parse_group(line: str, nano: nanoparticlebuilder.NanoparticleBuilder) -> None:
	# group		Ni type 2
	line = split_command(line)
	group_name = line[1]
	prop = line[2]
	if prop == "type":
		value = line[3]
		result = nano.add_group_type(value, group_name)
		assert is_correct_parsing(line, result), f"Group type {group_name} is not parsed correctly: \n{result} != \n{line}"
		logging.debug(result)
		return
	elif prop == "region":
		region_name = line[3]
		result = nano.add_group_region(region_name, group_name)
		logging.debug(result)
		return
	else:
		raise ValueError(f"Unknown group property: {prop}")


def parse_delete_atoms(line, nano):
	# delete_atoms region v compress yes
	line = split_command(line)
	selector_type = line[1]
	# Parse region
	if selector_type == "region":
		region_name = line[2]
		keywords = line[3:]  # [key1, value1, key2, value2, ...]
		# keywords = {keywords[i]: keywords[i + 1] for i in range(0, len(keywords), 2)}
		keywords = " ".join(keywords)
		result = nano.add_delete_atoms_region(region_name, keywords)
		logging.debug(result)
		return
	else:
		raise ValueError(f"Unknown selector type: {selector_type}")


def parse_line(line: str, nano: nanoparticlebuilder.NanoparticleBuilder) -> None:
	if line.startswith("#"):
		logging.debug(f"[blue]{line}[/blue]", extra={"markup": True})
	elif line.startswith("region"):
		logging.debug(f"[green]{line}[/green]", extra={"markup": True})
		parse_region(line, nano)
	elif line.startswith("create_atoms"):
		logging.debug(f"[magenta bold]{line}[/magenta bold]", extra={"markup": True})
		parse_create_atoms(line, nano)
	elif line.startswith("set"):
		logging.debug(f"[magenta]{line}[/magenta]", extra={"markup": True})
		parse_set(line, nano)
	elif line.startswith("group"):
		logging.debug(f"[grey]{line}[/grey]", extra={"markup": True})
		parse_group(line, nano)
	elif line.startswith("delete_atoms"):
		logging.debug(f"[red]{line}[/red]", extra={"markup": True})
		parse_delete_atoms(line, nano)

	else:
		raise ValueError(f"Unknown line: {line}")


def parse_shape(lines: list[str], nano: nanoparticlebuilder.NanoparticleBuilder) -> None:
	for line in lines:
		parse_line(line, nano)


def load_shapes(path: str, ignore: list[str]) -> dict[str, nanoparticlebuilder.NanoparticleBuilder]:
	for shape in sorted_recursive_input_search(path):
		if any([section in shape for section in ignore]):  # Ignore
			continue
		yield parse_single_shape(shape)


def parse_single_shape(shape: str, full_file: bool = False, replacements: dict | None = None) -> tuple[str, nanoparticlebuilder.NanoparticleBuilder]:
	"""
	Parses a single shape file
	:param shape:  Path to shape file
	:param full_file:  Whether to parse the full file or just the first region
	:return:
	"""
	replacements = replacements or {}
	with open(shape, "r") as f:
		logging.info(f"[yellow]=== {shape} ===[/yellow]", extra={"markup": True})
		nano = nanoparticlebuilder.NanoparticleBuilder(title=shape)
		lines = f.readlines()
		if not full_file:
			lines = locate_relevant_lines(lines)
		lines = [template.replace_templates(ll.strip(), replacements) for l in lines if (ll := l.strip()) != ""]
		parse_shape(lines, nano)
		return shape, nano


def locate_relevant_lines(lines):
	start = first_index_that_startswith(lines, "lattice") + 1
	try:
		end = first_index_that_startswith(lines, "# setting")
	except AssertionError:
		end = first_index_that_startswith(lines, "mass")
	return lines[start:end]


def first_index_that_startswith(lines: list[str], start: str) -> int:
	out = [i for i, l in enumerate(lines) if l.startswith(start)]
	assert len(out) > 0, f"Could not find line that starts with {start}"
	return out[0]
