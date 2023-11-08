import os
import re
import nanoparticle

import shapes as s
DO_PARSER_LOGGING = True

def recursive_input_search(path):
	output = []
	for file in os.listdir(path):
		if os.path.isdir(f"{path}/{file}"):
			output += recursive_input_search(f"{path}/{file}")
		elif file.endswith(".in"):
			output.append(f"{path}/{file}")
	return output


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
	else:
		log_output("\033[31mUnknown region type\033[0m")


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
	input_files = recursive_input_search(path)
	out_shapes = {}
	for shape in input_files:
		if any([section in shape for section in ignore]): # Ignore
			continue
		with open(shape, "r") as f:
			log_output(f"\033[33m=== {shape} ===\033[0m")
			nano = nanoparticle.Nanoparticle()
			lines = f.readlines()
			start = [i for i, l in enumerate(lines) if l.startswith("lattice")][0] + 1
			end = [i for i, l in enumerate(lines) if l.startswith("# setting")][0]
			lines = [ll.strip() for l in lines[start:end] if (ll := l.strip()) != ""]
			parse_shape(lines, nano)
			out_shapes[shape] = nano
	return out_shapes
