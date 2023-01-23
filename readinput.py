import numpy as np
import re
import sys
import other_functions as OF

def check_pi(string_data):
	""" Function that checks if the input data is pi or not """
	if string_data == 'pi':
		return np.pi
	elif string_data == "-pi":
		return -np.pi
	elif 'e' in string_data:
		return float(string_data)
	else:
		return np.float64(string_data)

def text_data_convert(text_data):
	if type(text_data) == np.ndarray:
		text_data = text_data[0]
	# Removes parentheses from the numbers
	if "("in text_data and ")" in text_data:
		text_data = re.sub("[()]", "", text_data)
	# If there is a division, split that up to numerator and denominator
	if "/" in text_data:
		numerator, denominator = text_data.split("/")
		# Do multiplication in numerator or denominator if they exist
		if "*" in numerator:
			elements = numerator.split("*")
			total1 = 1.0
			for i in range(len(elements)):
				total1 *= check_pi(elements[i])
		else:
			total1 = check_pi(numerator)
		if "*" in denominator:
			elements = denominator.split("*")
			total2 = 1.0
			for i in range(len(elements)):
				total2 *= check_pi(elements[i])
		else:
			total2 = check_pi(denominator)
		final_value = total1/total2
		# If there is no division but multiplication, split up and multiplies the elements
	elif "*" in text_data:
		elements = text_data.split("*")
		final_value = 1.0
		for i in range(len(elements)):
			final_value *= check_pi(elements[i])
	else:
		final_value = check_pi(text_data)
	return final_value

def set_global_params(r_data, tolerance_vars, time_vars, Main_params_confirmation):
	if r_data[0] == 't_end:':
		time_vars[0] = r_data[1].astype(np.float64)
		Main_params_confirmation[0] = 1
	elif r_data[0] == 'N:':
		time_vars[1] = r_data[1].astype(np.int64)
		Main_params_confirmation[1] = 1
	elif r_data[0] == 'ntol:':
		tolerance_vars[0] = r_data[1].astype(np.float64)
		Main_params_confirmation[2] = 1
	elif r_data[0] == 'iabstol:':
		tolerance_vars[1] = r_data[1].astype(np.float64)
		Main_params_confirmation[3] = 1
	elif r_data[0] == 'ireltol:':
		tolerance_vars[2] = r_data[1].astype(np.float64)
		Main_params_confirmation[4] = 1
	elif r_data[0] == 'quadval:':
		tolerance_vars[3] = r_data[1].astype(np.float64)
		Main_params_confirmation[5] = 1
	elif r_data[0] == 'hmax:':
		tolerance_vars[4] = r_data[1].astype(np.float64)
		Main_params_confirmation[6] = 1
	elif r_data[0] == 'hmin:':
		tolerance_vars[5] = r_data[1].astype(np.float64)
		Main_params_confirmation[7] = 1
	elif r_data[0] == "kepler:":
		tolerance_vars[6] = r_data[1].astype(np.float64)
		Main_params_confirmation[8] = 1
	elif r_data[0] == "sun:":
		Main_params_confirmation[9] = 1
		tolerance_vars[7] = r_data[1].astype(np.int32)
	elif r_data[0] == "density:":
		Main_params_confirmation[10] = 1
		tolerance_vars[8] = r_data[1].astype(np.int32)

def check_main_params(params_list, params_confirmation):
	Non_confirmed_params = np.where(params_confirmation == 0)[0]
	if Non_confirmed_params.size > 0:
		for i in Non_confirmed_params:
			print("Missing parameter " + params_list[i])
		sys.exit("Parameters are missing! Maybe you forgot semicolon : behind the parameter(s)?")

def array_size_check(data_arr, size, name, body_count):
	if len(data_arr) != size:
		sys.exit("Error: Number of elements for section %s is %d for body %d. Should be %d." 
				 %(name, len(data_arr), body_count, size))

def mass_scaling(masses):
	""" 
	Used to rescale the masses of the bodies and the gravitational constant.
	The rescaling factor will be the exponent of the largest mass.
	E.g. M = [5.47*1e30, 1.0*1e18, 2.0*1e19], the scaling will be
	scale = 1e30
	Masses will then be
	M_scaled = M/scale
	"""
	mass_exponents = np.floor(np.log10(np.abs(masses)))
	return np.max(mass_exponents)


def read_input_data(filename):
	input_file = open(filename, "r")
	time_vars = [0.0, 0]
	tolerance_vars = np.zeros(9)
	velocities = []
	positions = []
	ang_speed = []
	angles = []
	masses = []
	semiaxes = []

	filenames = []
	vertices = np.array([])
	face_ids = np.array([], dtype=np.int)
	N_vertices = []
	N_faces = []

	binary_ids = []
	Main_params_list = ['t_end:', 'N:', 'ntol:', 'iabstol:', 'ireltol:',
						'quadval:', 'hmax:', 'hmin:', 'kepler:', 'sun:', 'density:']
	Main_params_confirmation = np.zeros(len(Main_params_list))
	Main_params_check_counter = 0
	get_data = 0
	body_count = 0
	Global_params = 0   # Ensures to check the first eight parameters.
	for line in input_file:
		r_data = np.array(line.split())
		N_data = len(r_data)
		if r_data.size == 0:
			# Skip empty lines
			continue
		else:
			if (r_data == '#').any() or (r_data == '###').any() or "#" in r_data[0]:
				# Skip lines with a # symbol
				continue
			else:
				# Reads and sorts data
				set_global_params(r_data, tolerance_vars, time_vars, Main_params_confirmation)
				if Global_params > len(Main_params_list)-1:
					if Main_params_check_counter == 0:
						check_main_params(Main_params_list, Main_params_confirmation)
						Main_params_check_counter = 1
					if r_data[0] == "@":
						get_data += 1
						body_count = 0
					else:
						if get_data == 1:
							masses.append(text_data_convert(r_data))
						elif get_data == 2:
							if len(r_data) == 3:
								# An ellipsoids
								semiaxes.append([text_data_convert(r_data[0]), text_data_convert(r_data[1]), text_data_convert(r_data[2])])
								filenames.append('')
								N_vertices.append(0)
								N_faces.append(0)
							elif len(r_data) == 1:
								# A polyhedron
								vertices_b, face_ids_b, N_vertices_b, N_faces_b = read_obj_file(r_data[0])
								filenames.append(r_data[0])
								vertices = np.concatenate([vertices, vertices_b])
								#face_ids.append(face_ids_b)
								face_ids = np.concatenate([face_ids, face_ids_b])
								N_vertices.append(N_vertices_b)
								N_faces.append(N_faces_b)
								semiaxes.append([0, 0, 0])
							elif len(r_data) == 2:
								# A polyhedron, but also includes size scaling
								vertices_b, face_ids_b, N_vertices_b, N_faces_b = read_obj_file(r_data[0])
								filenames.append(r_data[0])
								vertices = np.concatenate([vertices, vertices_b*float(r_data[1])])
								face_ids = np.concatenate([face_ids, face_ids_b])
								N_vertices.append(N_vertices_b)
								N_faces.append(N_faces_b)
								semiaxes.append([0, 0, 0])
							else:
								sys.exit("Number of semiaxes elements not correct for body %d" %body_count)
						elif get_data == 3:
							array_size_check(r_data, 3, "Positions", body_count)
							positions.append([text_data_convert(r_data[0]), text_data_convert(r_data[1]), text_data_convert(r_data[2])])
						elif get_data == 4:
							array_size_check(r_data, 3, "Velocities", body_count)
							velocities.append([text_data_convert(r_data[0]), text_data_convert(r_data[1]), text_data_convert(r_data[2])])
						elif get_data == 5:
							array_size_check(r_data, 3, "Angles", body_count)
							angle1 = text_data_convert(r_data[0])
							angle2 = text_data_convert(r_data[1])
							angle3 = text_data_convert(r_data[2])
							angles.append([np.deg2rad(angle1), np.deg2rad(angle2), np.deg2rad(angle3)])
						elif get_data == 6:
							array_size_check(r_data, 3, "Angular velocities", body_count)
							ang_speed.append([text_data_convert(r_data[0]), text_data_convert(r_data[1]), text_data_convert(r_data[2])])
						elif get_data == 7:
							binary_ids.append(int(r_data[0]))
						body_count += 1
					
				Global_params += 1
	if get_data < 6:
		print("Missing data parameter")
		print("Check that masses, semiaxes, positions, velocities, angles and angular velocities are included")
		print("Did you remove the @ lines in the input file?")
		sys.exit(0)
	if get_data < 7 and tolerance_vars[6] > 0: 
		print("Missing data parameter")
		print("Check that masses, semiaxes, positions, velocities, angles, angular velocities and BIDs are included")
		print("Did you remove the @ lines in the input file?")
		sys.exit(0)
		
	input_file.close()
	masses = np.array(masses)
	semiaxes = np.array(semiaxes, dtype=np.double)
	velocities = np.array(velocities)
	positions =  np.array(positions)
	ang_speed = np.array(ang_speed)
	angles = np.array(angles)
	face_ids = np.array(face_ids, dtype=np.int32)
	N_vertices = np.array(N_vertices)
	N_faces = np.array(N_faces)
	N_bodies = len(masses)
	
	if (N_vertices > 0).any() and len(N_vertices) > 2:
		sys.exit("The code is restricted to a two-body problem for any polyhedral shapes")
	if time_vars[1] <= 0:
		sys.exit("Error: Number of computation points N must be larger than 0.")
	if (masses < 0).any():
		sys.exit("Error: Masses cannot be zero or negative.")
	if np.size(masses) < 2:
		sys.exit("Error: There are less than 2 ellipsoids in the input file!")
	if tolerance_vars[6] != 0 and tolerance_vars[6] != 1 and tolerance_vars[6] != 2:
		sys.exit("Error: kepler: must take value 0, 1 or 2.")
	for i in range(N_bodies):
		all_zero = (semiaxes[i] == 0).all()
		not_zero = (semiaxes[i] > 0).all()
		if not (all_zero or not_zero):
			sys.exit("Error: Semiaxes must all either be zero or larger than zero.")

	if len(semiaxes) != N_bodies:
		sys.exit("Error: Not enough bodies in the semiaxes section.")
	if len(velocities) != N_bodies:
		sys.exit("Error: Not enough bodies in the velocities section.")
	if len(positions) != N_bodies:
		sys.exit("Error: Not enough bodies in the positions section.")
	if len(ang_speed) != N_bodies:
		sys.exit("Error: Not enough bodies in the angular velocity section.")
	if len(angles) != N_bodies:
		sys.exit("Error: Not enough bodies in the angles section.")
	if len(binary_ids) != N_bodies and tolerance_vars[6] > 0:
		sys.exit("Error: Not enough binary ids in the BID section.")
	
	# Density tag enabled. Converts density to mass. Only applicable for ellipsoids
	vcount = 0
	fcount = 0
	if tolerance_vars[8] == 1:
		for i in range(N_bodies):
			if N_vertices[i] > 0:
				vertices_in = vertices[vcount:vcount+3*N_vertices[i]]
				faces_in = face_ids[fcount:fcount+3*N_faces[i]]
				v_new, volume = OF.polyhedron_volume_centroid_single(vertices_in, N_vertices[i], N_faces[i], faces_in)
				vertices[vcount:vcount+3*N_vertices[i]] = v_new
				vcount += 3*N_vertices[i]
				fcount += 3*N_faces[i]
				masses[i] = masses[i]*volume
			else:
				if semiaxes[i].any():
					masses[i] = masses[i]*4*np.pi*semiaxes[i][0]*semiaxes[i][1]*semiaxes[i][2]/3
				else:
					sys.exit("Density cannot be converted to mass for spheres with zero radius.")
	tspan = np.linspace(0.0, time_vars[0], time_vars[1])
	y_init = np.concatenate([velocities.ravel(), positions.ravel(), ang_speed.ravel(), angles.ravel()])
	scaling_exponent = mass_scaling(masses)
	if scaling_exponent < 1e-15:
		scaling_exponent = 0.0
	if tolerance_vars[7] == 1:
		mass_scale = 1.989*1e30
	elif tolerance_vars[7] == 0:
		mass_scale = 10**scaling_exponent
	else:
		sys.exit("Error: sun: must take values 0 or 1.")
	
	masses /= mass_scale
	return_list = [tspan, masses, y_init, semiaxes, N_bodies, tolerance_vars, mass_scale,
				   binary_ids, vertices.ravel(), face_ids, N_vertices, N_faces, filenames]
	return return_list


def read_obj_file(filename):
	"""
	Reads and sorts polyhedron .obj file.
	"""
	if not ".obj" in filename:
		filename = filename + ".obj"
	data = open("polyhedrondata/"+filename, 'r')
	vertices = []
	face_ids = []
	for line in data:
		stuff = line.split()
		if stuff[0] == 'v':
			vertices.append([float(stuff[1]), float(stuff[2]), float(stuff[3])])
		if stuff[0] == 'f':
			face_ids.append([int(stuff[1])-1, int(stuff[2])-1, int(stuff[3])-1])

	N_vertices = len(vertices)
	N_faces = len(face_ids)

	vertices = np.array(vertices)
	face_ids = np.array(face_ids)

	return vertices.ravel(), face_ids.ravel(), N_vertices, N_faces

if __name__ == '__main__':
	# For testing purposes
	read_input_data("initial_values/example.txt")