import os
import re
import nanoparticle

import shapes as s

DO_PARSER_LOGGING = False


def recursive_input_search(path):
	for file in os.listdir(path):
		if os.path.isdir(f"{path}/{file}"):
			yield from recursive_input_search(f"{path}/{file}")
		elif file.endswith(".in"):
			yield f"{path}/{file}"


def parse_region(line, nano):
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
		full_length = abs(pos_length - neg_length)
		coord_c = (pos_length + neg_length) / 2.0
		shape = None
		if axis == "x":
			shape = s.Cylinder(radius, full_length, axis, (coord_c, coord_a, coord_b), check_in_box=False)
		elif axis == "y":
			shape = s.Cylinder(radius, full_length, axis, (coord_a, coord_c, coord_b), check_in_box=False)
		elif axis == "z":
			shape = s.Cylinder(radius, full_length, axis, (coord_a, coord_b, coord_c), check_in_box=False)
		assert assert_correct_parsing(line, shape.get_region(region_name)), f"Region {region_name} is not parsed correctly: \n{split_command(shape.get_region(region_name))} != \n{line}"
		nano.add_named_shape(shape, region_name)
		log_output(shape)
		return shape
	elif region_type == "sphere":
		# region sphere 0 0 0 10 units box
		coord_a = float(region_args[0])
		coord_b = float(region_args[1])
		coord_c = float(region_args[2])
		radius = float(region_args[3])
		shape = s.Sphere(radius, (coord_a, coord_b, coord_c))
		assert assert_correct_parsing(line, shape.get_region(region_name)), f"Region {region_name} is not parsed correctly: \n{split_command(shape.get_region(region_name))} != \n{line}"
		nano.add_named_shape(shape, region_name)
		log_output(shape)
		return shape
	elif region_type == "plane":
		# region plane 0 0 0 0 0 1 units box
		coord_a = float(region_args[0])
		coord_b = float(region_args[1])
		coord_c = float(region_args[2])
		normal_a = float(region_args[3])
		normal_b = float(region_args[4])
		normal_c = float(region_args[5])
		shape = s.Plane((coord_a, coord_b, coord_c), (normal_a, normal_b, normal_c))
		assert assert_correct_parsing(line, shape.get_region(region_name)), f"Region {region_name} is not parsed correctly: \n{split_command(shape.get_region(region_name))} != \n{line}"
		nano.add_named_shape(shape, region_name)
		log_output(shape)
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
		shape = s.Cone(axis, coord_a, coord_b, radlo, radhi, lo, hi)
		assert assert_correct_parsing(line, shape.get_region(region_name)), f"Region {region_name} is not parsed correctly: \n{split_command(shape.get_region(region_name))} != \n{line}"
		nano.add_named_shape(shape, region_name)
		log_output(shape)
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
		shape = s.Prism(xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz)
		assert assert_correct_parsing(line, shape.get_region(region_name)), f"Region {region_name} is not parsed correctly: \n{split_command(shape.get_region(region_name))} != \n{line}"
		nano.add_named_shape(shape, region_name)
		log_output(shape)
		return shape
	elif region_type == "intersect":
		# intersect args = N reg-ID1 reg-ID2 ...
		#   N = # of regions to follow, must be 2 or greater
		#   reg-ID1,reg-ID2, ... = IDs of regions to intersect

		#  region_args = ['2', 'sq', 'ce', 'units', 'box']
		n = int(region_args[0])
		reg_ids = region_args[1:1+n]
		nano.add_intersect(reg_ids, region_name)
	else:
		log_output("\033[31mUnknown region type\033[0m")
		raise ValueError(f"Unknown region type: {region_type}")


def assert_correct_parsing(line, command):
	parsed_command = split_command(command)
	for x in range(0, len(parsed_command)):
		if (parsed_command[x] != line[x]
			and f"{parsed_command[x]}.0" != line[x]
			and f"{line[x]}.0" != parsed_command[x]
			and abs(float(parsed_command[x]) - float(line[x])) > 0.0001):
			log_output(f"\033[31m{parsed_command[x]} != {line[x]}\033[0m")
			return False
	return True


def parse_create_atoms(line, nano):
	# Remove multiple spaces
	line = split_command(line)
	atom_type = line[1]
	selector_type = line[2]
	selector_args = line[3:]
	# Parse region
	if selector_type == "region":
		result = nano.add_create_atoms(atom_type, selector_args[0])
		log_output(">> " + result)
		return
	else:
		log_output("\033[31mUnknown selector type\033[0m")
		raise ValueError(f"Unknown selector type: {selector_type}")


def split_command(line):
	return re.split("\\s+", line)


def log_output(param):
	global DO_PARSER_LOGGING
	if DO_PARSER_LOGGING:
		print(param)


def parse_set(line, nano):
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
			log_output(">>" + result)
			return
		elif prop == "type/subset":
			value = set_args[2]
			count = set_args[3]
			seed = set_args[4]
			result = nano.add_set_type_subset_region(value, region_name, count, seed)
			log_output(">>" + result)
			return
		else:
			log_output("\033[31mUnknown set property\033[0m")
			raise ValueError(f"Unknown set property: {prop}")
	elif set_type == "group":
		group_name = set_args[0]
		prop = set_args[1]
		if prop == "type/subset":
			value = set_args[2]
			count = set_args[3]
			seed = set_args[4]
			result = nano.add_set_type_subset_group(value, group_name, count, seed)
			assert assert_correct_parsing(line, result), f"Set type subset group {group_name} is not parsed correctly: \n{result} != \n{line}"
			log_output(">>" + result)
			return
		elif prop == "type/ratio":
			value = set_args[2]
			ratio = set_args[3]
			seed = set_args[4]
			result = nano.add_set_type_ratio_group(value, group_name, ratio, seed)
			assert assert_correct_parsing(line, result), f"Set type ratio group {group_name} is not parsed correctly: \n{result} != \n{line}"
			log_output(">>" + result)
			return
		else:
			log_output("\033[31mUnknown set property\033[0m")
			raise ValueError(f"Unknown set property: {prop}")
	else:
		log_output("\033[31mUnknown selector type\033[0m")
		raise ValueError(f"Unknown selector type: {set_type}")


def parse_group(line, nano):
	# group		Ni type 2
	line = split_command(line)
	group_name = line[1]
	prop = line[2]
	if prop == "type":
		value = line[3]
		result = nano.add_group_type(value, group_name)
		assert assert_correct_parsing(line, result), f"Group type {group_name} is not parsed correctly: \n{result} != \n{line}"
		log_output(">>" + result)
		return


def parse_line(line, nano):
	if line.startswith("#"):
		log_output(f"\033[34m{line}\033[0m")
	# Is comment
	elif line.startswith("region"):
		log_output(f"\033[32m{line}\033[0m")
		parse_region(line, nano)
	elif line.startswith("create_atoms"):
		log_output(f"\033[35;1m{line}\033[0m")
		parse_create_atoms(line, nano)
	elif line.startswith("set"):
		log_output(f"\033[35m{line}\033[0m")
		parse_set(line, nano)
	elif line.startswith("group"):
		log_output(f"\033[2;37m{line}\033[0m")
		parse_group(line, nano)
	else:
		raise ValueError(f"Unknown line: {line}")


def parse_shape(lines, nano):
	for line in lines:
		parse_line(line, nano)


def load_shapes(path, ignore) -> dict[str, nanoparticle.Nanoparticle]:
	for shape in recursive_input_search(path):
		if any([section in shape for section in ignore]):  # Ignore
			continue
		yield parse_single_shape(shape)


def parse_single_shape(shape: str):
	with open(shape, "r") as f:
		log_output(f"\033[33m=== {shape} ===\033[0m")
		nano = nanoparticle.Nanoparticle({'title': shape})
		lines = f.readlines()
		start = first_index_that_startswith(lines, "lattice") + 1
		try:
			end = first_index_that_startswith(lines, "# setting")
		except AssertionError:
			end = first_index_that_startswith(lines, "mass")
		lines = [ll.strip() for l in lines[start:end] if (ll := l.strip()) != ""]
		parse_shape(lines, nano)
		return shape, nano


def first_index_that_startswith(lines, start):
	out = [i for i, l in enumerate(lines) if l.startswith(start)]
	assert len(out) > 0, f"Could not find line that starts with {start}"
	return out[0]
