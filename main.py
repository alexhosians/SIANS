"""
Main python file that runs the whole software.
See documentation for example usage.
"""
## Basic python modules
import time, os
import argparse
import shutil
import numpy as np
## Other modules
import readinput
import readsolution
import other_functions as OF

#########################################################################

# Defining some global pre-factors
AU = 149597871  # km
# Date+time based folder used to save 
runtime = time.strftime("%Y/%m_%d/")
plot_save_folder = "PlotResults/"
Folder_location = os.path.dirname(os.path.realpath(__file__))

################################################################

def boolean_check(textarg):
	if textarg.lower() in ('true', 't', 'yes', 'y', '1'):
		return True
	elif textarg.lower() in ('false', 'f', 'no', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError("Boolean input expected.")

def Argument_parser():
	""" Parses optional argument when program is run from the command line """
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	# Optional arguments
	parser.add_argument("-outfold", "--FolderExtra",\
						help="Adds an additional subfolder for different plot results \nof the same date. "\
						" Overrides the input filename if used.",
			 			type=str, default=None)
	parser.add_argument("-delsol", "--DeletePickle",\
						help="If True, Deletes the solution files of the output folder \n(assuming it exist).",
						type=boolean_check, default=False)
	parser.add_argument("-rundate", "--RunDate",\
						help="Specifies which folder date is to be saved/read from. \nInput can be 20yy/mm_dd/ or any string."\
						"\nRuntime date by default.",
						type=str, default=time.strftime("%Y/%m_%d/"))
	parser.add_argument("-method", "--ODESolver",\
						help="Selects numerical integrator to be used as the solver.\nAvailable options are (-input -- description): \n" \
							"-RK4    -- Classical Runge-Kutta method of order 4 \n"\
							"-RK54   -- Runge-Kutta of order 5(4) method (DEFAULT) \n" \
							"-DP89   -- Dormand-Prince Runge-Kutta of order 8(9) method \n"\
							"-V65	-- Verner Runge-Kutta method of order 6(5)\n"\
							"-V76	-- Verner Runge-Kutta method of order 7(6)\n"\
							"-V87	-- Verner Runge-Kutta method of order 8(7)\n"\
							"-V98	-- Verner Runge-Kutta method of order 9(8)\n"\
							"-T98	-- Tsitouras Runge-Kutta method of order 9(8) \n"\
							"-F108   -- Feagin Runge-Kutta method of order 10(8) \n"\
							"-F1210  -- Feagin Runge-Kutta method of order 12(10) \n",
						type=str, default="RK54")
	parser.add_argument("-input", "--InputFile",\
						help="Filename (without .txt) of the input file. The input file \ncontains the global parameters,"\
						" e.g. runtime, and ellipsoid \nparameters. "\
						"If not used, input parameters are used within \n the code, which is manually set. \n"\
						"The save folder will be named after the input file \nHowever, if the argument -foldExtra is used "\
						"the name from \nthe -foldExtra argument will be used.",
						type=str, default="example")
	parser.add_argument("-units", "--useGgrav", \
						help="If set to SI, sets: \n   G_grav = 6.674e-11 m^3/(kg s^2). \n"\
						"If set to kmhr, sets: \n   G_grav = (3600^2/1000^3)*6.674e-11 km^3/(kg hr^2) \n"\
						"If set to kms, sets: \n   G_grav = (1/1000^3)*6.674e-11 km^3/(kg s^2) \n"\
						"If none, sets G = 1.0 (DEFAULT)",
						type=str, default=None)
	parser.add_argument("-adaptive", "--useAdaptive",\
						help="If False, does not utilize adaptive time stepper. \nDefault: True.",\
						type=boolean_check, default=True)
	parser.add_argument("-euler", "--UseEulerParams",
						help="If True, converts Tait-Bryan angles \nto Euler angles and uses them\n"
							 "to compute the rotational motion of the bodies. \nDefault: False",
						type=boolean_check, default=False)
	parser.add_argument("-mpir", "--UseMpir",
						help="If True, uses mpir implementation \nof polyhedron potential solver",
						type=boolean_check, default=False)
	parser.add_argument("-doplot", "--doplot", help="If True, plots basic data provided by the software", 
						type=boolean_check, default=False)
	parser.add_argument("-solconv", "--Convertsolution", help="If True, converts the Numpy solution \nfiles to a standard format."
						"\nDefault: False",
						type=boolean_check, default=False)
	parser.add_argument("-soltype", "--SolutionType", help="The format of the non-Numpy solution files."
						" \nCan be 'txt' (text) or 'csv' \nDefault: txt",
						type=str, default='txt')
	# Parse arguments
	args = parser.parse_args()
	# Checks if certain input parameters are valid
	method_checks = np.array(["RK4", "RK54", "V65", "V76", "V87", "V98", "F108", "F1210", "T98"])
	SI_checks = np.array(["SI", "kmhr", "kmd", "kmyr", "auyr", "aud", None])
	if not (args.ODESolver == method_checks).any():
		raise ValueError("Method %s is not valid. Try the following: \n"
			"-method RK4 \n"
			"-method RK54 \n"
			"-method V65 \n"
			"-method V76 \n"
			"-method V87 \n"
			"-method V98 \n"
			"-method T98 \n"
			"-method F108 \n"
			"-method F1210" %args.ODESolver)
	if not (args.useGgrav == SI_checks).any():
		raise ValueError("-units input %s not valid. Try the following: \n"
			"-units SI \n"
			"-units kmhr \n"
			"-units kms" %args.useGgrav)
	if args.InputFile and not args.FolderExtra:
		args.FolderExtra = args.InputFile
	if args.ODESolver == "RK4":
		# Defaults to non-adaptive method with 4th order Runge-Kutta method
		args.useAdaptive = False
	if args.InputFile:
		if ".txt" in args.InputFile:
			raise ValueError("Input filename %s should not contain .txt. Try: -input %s" %(args.InputFile, args.InputFile.replace(".txt","")))
		else:
			args.InputFile = "initial_values/" + args.InputFile + ".txt"
	else:
		args.InputFile = False

	if not args.SolutionType == 'txt' and not args.SolutionType == 'csv':
		raise ValueError("Solution filetype '%s' not recognized" %args.SolutionType)

	print('Run python code with -h argument to see extra optional arguments')
	return args


class SaveFigures():
	def __init__(self, results_dir, fileExtension):
		self.results_dir = results_dir
		self.fileExtension = fileExtension

	def savefigure(self, figure, name):
		""" Function that calls savefig based on figure instance and filename. """
		if type(name) != str:
			raise ValueError('filename not a string!')
		figure.savefig(self.results_dir + name + self.fileExtension, bbox_inches='tight')

class CythonToScipy:
	""" 
	A class that converts Cython solution instance to a class instance similar to scipy.
	"""
	def __init__(self, CySol, Nbods, Use_euler_params):
		self.nfev = CySol.nfev
		self.NSolFiles = CySol.NSolFiles
		self.N_bodies = Nbods
		self.crash = CySol.crash
		if self.NSolFiles == 1:
			self.t = CySol.t
			if Use_euler_params:
				nvars = 13
			else:
				nvars = 12
			self.var = np.zeros((Nbods*nvars,self.nfev))
			for i in range(Nbods*nvars):
				self.var[i] = CySol.var[i]
		else:
			self.t = 0
			self.var = 0

##############################################################

def Kepler2Cartesian(p1, p2, mu, time_convert, input_select):
	"""
	Converts Keplerian coordinates to Cartiesian coordinates. 
	Conversion follows: Orbital Mechanics for Engineering Students, by Howards Curtis.

	The input parameters are:
	D: orbital period (days)
	e: eccentricity
	i: inclination (degrees)
	RA: ascension node (degrees)
	omega: longitude of perihelion (degrees)
	l: mean longitude (degrees)
	"""
	e = p1[1]
	i = np.deg2rad(p1[2])
	RA = np.deg2rad(p2[0])
	omega = np.deg2rad(p2[1])
	if input_select == 1:
		D = p1[0]*time_convert
		l = np.deg2rad(p2[2])
		# Semimajor axis
		a = (mu*(D/(2*np.pi))**2)**(1.0/3.0)
		# Mean anomaly
		M_0 = l - omega
	elif input_select == 2:
		a = p1[0]
		M_0 = np.deg2rad(p2[2])
	else:
		raise ValueError("Kepler input must take values 1 or 2.")
	# Specific angular momentum
	h = np.sqrt(a*mu*(1-e**2))
	# True anomaly
	M = M_0 + (2*e-(e**3)/4.0)*np.sin(M_0) + 5*(e**2)*np.sin(2*M_0)/4.0 + 13*(e**3)*np.sin(3*M_0)/12.0 # +O(e^4)
	r_xt = (h**2/mu)*(1/(1+e*np.cos(M)))*np.array([np.cos(M), np.sin(M), 0.0])
	v_xt = (mu/h)*np.array([-np.sin(M), e+np.cos(M), 0.0])
	sRA = np.sin(RA)
	cRA = np.cos(RA)
	sOm = np.sin(omega)
	cOm = np.cos(omega)
	si = np.sin(i)
	ci = np.cos(i)
	Q = np.array([[cRA*cOm - sRA*sOm*ci, -cRA*sOm - sRA*ci*cOm, sRA*si],
				  [sRA*cOm + cRA*sOm*ci, -sRA*sOm + cRA*ci*cOm, -cRA*si],
				  [si*sOm, si*cOm, ci]])
	r_pos = np.dot(Q, r_xt)
	v_pos = np.dot(Q, v_xt)
	return np.concatenate([r_pos, v_pos])

def TB_to_euler_params(TB_angles):
	""" Converts Tait-Bryan angles to Euler parameters """
	R = OF.rotation_matrix(TB_angles[0], TB_angles[1], TB_angles[2])
	trace_R = np.trace(R)
	e_sq_array = np.zeros(4)
	e_sq_array[0] = (trace_R + 1)/4

	Euler_params = np.zeros(4)
	if e_sq_array[0] != 0:
		# If e_0 /= 0
		Euler_params[0] = np.sqrt(e_sq_array[0])
		Euler_params[1] = (R[2][1] - R[1][2])/(4*Euler_params[0])
		Euler_params[2] = (R[0][2] - R[2][0])/(4*Euler_params[0])
		Euler_params[3] = (R[1][0] - R[0][1])/(4*Euler_params[0])
	else:
		# If e_0 = 0
		for i in range(3):
			e_sq_array[i+1] = (1 + 2*R[i][i] - trace_R)/4
		if e_sq_array[1] != 0:
			Euler_params[1] = np.sqrt(e_sq_array[1])
			Euler_params[2] = (R[1][0] + R[0][1])/(4*Euler_params[1])
			Euler_params[3] = (R[2][0] + R[0][2])/(4*Euler_params[1])
		elif e_sq_array[2] != 0:
			Euler_params[2] = np.sqrt(e_sq_array[2])
			Euler_params[1] = (R[1][0] + R[0][1])/(4*Euler_params[2])
			Euler_params[3] = (R[2][1] + R[1][2])/(4*Euler_params[2])
		elif e_sq_array[3] != 0:
			Euler_params[3] = np.sqrt(e_sq_array[3])
			Euler_params[0] = (R[2][0] + R[0][2])/(4*Euler_params[3])
			Euler_params[2] = (R[2][1] + R[1][2])/(4*Euler_params[3])
		else:
			raise ValueError("No conditions met for Euler parameter conversion!")
	# Check constraint of euler parameters
	Euler_condition = 0
	for i in range(4):
		Euler_condition += Euler_params[i]**2
	if np.abs((Euler_condition - 1))  > 1e-12:
		raise ValueError("Sum of euler parameters = %.5f. Does not satisfy constraint." %Euler_condition)
	return Euler_params

def TB_to_Euler_params_init(y_init, N_bodies):
	""" 
	Function that converts y_init array with Tait-Bryan angles
	to an array that with Euler parameters.
	With Tait-Bryan angles, each body has 12 parameters.
	With Euler parameters, each body has 13 parameters.
	"""
	y_init_new = np.zeros(N_bodies*13)
	y_init_new[:9*N_bodies] = y_init[:9*N_bodies]
	for i in range(N_bodies):
		y_init_new[9*N_bodies+i*4:9*N_bodies+(i+1)*4] = TB_to_euler_params(y_init[9*N_bodies+i*3:9*N_bodies+(i+1)*3])
	return y_init_new

##############################################################

def default_tolerance_check(tolerance_array, method):
	"""
	Function that sets default numerical integration tolerance ntol.
	This tolerance is used for adaptive time stepping and not quadrature integration!
	If ntol is zero, sets ntol to default, which varies based on numerical integrators.
	Does nothing if ntol is not zero.
	"""
	if tolerance_array[0] == 0:
		if method == "RK54":
			tolerance_array[0] = 1e-8
		elif method == "V65":
			tolerance_array[0] = 1e-9
		elif method == "V76":
			tolerance_array[0] = 1e-10
		elif method == "V87":
			tolerance_array[0] = 1e-12
		elif method == "V98":
			tolerance_array[0] = 1e-13
		elif method == "T98":
			tolerance_array[0] = 1e-13
		elif method == "F108":
			tolerance_array[0] = 1e-12
		elif method == "F1210":
			tolerance_array[0] = 1e-12

def Set_G_parameter(semiaxes, input_arg):
	""" Converts the gravitational constant based on unit input """
	time_convert = 0
	E_convert = 1  # Used to convert energy unit to J/mass_scale
	unit_text = ['','']
	if input_arg == "SI":
		G_g = 6.67408e-11
		time_convert = 24*60*60
		E_convert = 1
		unit_text = ['m', 's']
	elif input_arg == "kmhr":
		G_g = 6.67408e-11*(3600**2/1000**3)
		time_convert = 24
		E_convert = (1000**2/3600**2)
		unit_text = ['km', 'hrs']
	elif input_arg == "kms":
		G_g = 6.67408e-11*(1/1000**3)
		time_convert = 1
		E_convert = (1000**2)
		unit_text = ['km', 's']
	else:
		G_g = 1.0
	return G_g, time_convert, E_convert, unit_text, semiaxes

def Orbel_initial_conditions(y_init, N_bodies, masses, binary_ids, G_g, tolarr, time_convert, mass_scale):
	""" 
	Sets cartesian initial conditions based on initial orbital elements 
	For two bodies and no sun, assumes that the primary is at rest at the origin
	For two bodies with sun, calculates as normal
	"""
	N3 = N_bodies*3
	N6 = N_bodies*6
	N9 = N_bodies*9
	if N_bodies == 2:
		if tolarr[7] == 1:
			# Sun is included
			mu_prim = G_g*(masses[0] + 1)
			mu_sec = G_g*(masses[0] + masses[1])
			mu_vals = [mu_prim, mu_sec]
			for i in range(2):
				i3 = i*3
				posvels = Kepler2Cartesian(y_init[i3:i3+3], y_init[N3+i3:N3+i3+3], mu_vals[i], time_convert, tolarr[6])
				# Positions
				y_init[i3:i3+3] = posvels[3:]
				# Velocities
				y_init[N3+i3:N3+i3+3] = posvels[:3]
				# Angles
				y_init[N9+i3:N9+i3+3] = y_init[N9+i3:N9+i3+3]
				if i == 1:
					y_init[i3:i3+3] += y_init[0:3]
					y_init[N3+i3:N3+i3+3] += y_init[6:9]

		else:
			# Sun is excluded
			mu = G_g*np.sum(masses)
			for i in range(2):
				i3 = i*3
				if i == 0:
					posvels = np.zeros(6)
				else:
					posvels = Kepler2Cartesian(y_init[i3:i3+3], y_init[N3+i3:N3+i3+3], mu, time_convert, tolarr[6])
				# Positions
				y_init[i3:i3+3] = posvels[3:]
				# Velocities
				y_init[N3+i3:N3+i3+3] = posvels[:3]
				# Angles
				y_init[N9+i3:N9+i3+3] = y_init[N9+i3:N9+i3+3]
	else:
		primary_mass = 0
		for i in range(N_bodies):
			i3 = i*3
			if tolarr[7] == 1:
				if binary_ids[i] == 0:
					primary_mass = masses[i]
					mu = G_g*(primary_mass + 1)
				else:
					if i == 0:
						raise ValueError("First element must be a primary body.")
					else:
						mu = G_g*(primary_mass + masses[i])
			else:
				if N_bodies > 2:
					if i == 0: # Assumes sun to be element 0
						mu = G_g*masses[i]
					else:
						mu = G_g*(masses[i] + masses[i-1])
				else:
					mu = G_g*(masses[1] + masses[0])
			posvels = Kepler2Cartesian(y_init[i3:i3+3], y_init[N3+i3:N3+i3+3], mu, time_convert, tolarr[6])
			# Positions
			y_init[i3:i3+3] = posvels[3:]
			# Velocities
			y_init[N3+i3:N3+i3+3] = posvels[:3]
			# Angles
			y_init[N9+i3:N9+i3+3] = y_init[N9+i3:N9+i3+3]
			# Rotate global angular velocity to local (maybe input local angular vel?)
			R = OF.rotation_matrix(y_init[N9+i3], y_init[N9+i3+1], y_init[N9+i3+2])
			y_init[N6+i3:N6+i3+3] = np.dot(R, y_init[N6+i3:N6+i3+3])
			if tolarr[7] == 1:
				if i > 0 and masses[i] < primary_mass:
					y_init[N3+i3:N3+i3+3] += y_init[N3:N3+3]
					y_init[i3:i3+3] += y_init[0:3]
			else:
				if i > 1 and mass_scale > 1e29:
					# Assumes sum in center
					y_init[N3+i3:N3+i3+3] += y_init[N3+3:N3+3+3]
					y_init[i3:i3+3] += y_init[3:6]
				elif i > 0 and mass_scale < 1e29:
					# Assumes no sun
					y_init[N3+i3:N3+i3+3] += y_init[N3:N3+3]
					y_init[i3:i3+3] += y_init[0:3]
	return y_init


def write_out_param_data(Textfile_out, t_array, masses, y_init, semiaxes, N_bodies, 
						tolarr, units, vert_filenames, parsed_arguments):
	"""
	Writes an output textfile that serves as a receit on the input values
	used in the simulation.
	"""
	N3 = N_bodies*3
	N6 = N_bodies*6
	N9 = N_bodies*9
	if not units[0]:
		distance_unit = "Dimensionless"
		mass_unit = ""
	else:
		distance_unit = units[0]
	if not units[1]:
		time_unit = "Dimensionless"
	else:
		time_unit = units[1]
	if tolarr[8] >= 1:
		mass_unit = "kg/%s^3" %distance_unit
	else:
		mass_unit = "kg"

	with open(Textfile_out, 'a') as text_file:
		text_file.write("Input data for this simulation \n")
		text_file.write("Name of input file: %s \n \n" %parsed_arguments.InputFile)
		
		text_file.write("== Global parameters: \n")
		text_file.write("t_end: %.1f \n" %t_array[-1])
		text_file.write("N: %d \n" %len(t_array))
		text_file.write("ntol: %.1e \n" %tolarr[0])
		text_file.write("iabstol: %.1e \n" %tolarr[1])
		text_file.write("ireltol: %.1e \n" %tolarr[2])
		text_file.write("quadval: %d \n" %tolarr[3])
		text_file.write("hmax: %.1e \n" %tolarr[4])
		text_file.write("hmin: %.1e \n" %tolarr[5])
		text_file.write("kepler: %d \n" %tolarr[6])
		text_file.write("sun: %d \n" %tolarr[7])
		text_file.write("density: %d \n \n" %tolarr[8])
		text_file.write("Distance unit: %s \n" %distance_unit)
		text_file.write("Time unit: %s \n" %time_unit)
		if tolarr[6] > 0:
			text_file.write("Angle units: degrees \n")
		else:
			text_file.write("Angle units: radians \n")
		text_file.write("Method: %s \n" %parsed_arguments.ODESolver)
		if parsed_arguments.useAdaptive:
			text_file.write("Adaptive time stepper: On \n \n")
		else:
			text_file.write("Adaptive time stepper: Off \n \n")
			
		text_file.write("For the variables, each line correspond to the body as below: \n")
		
		text_file.write("== Semiaxes (a, b, c): \n")
		for i in range(N_bodies):
			if vert_filenames[i]:
				text_file.write("%s \n" %vert_filenames[i])
			else:
				text_file.write("(%.3f, %.3f, %.3f) \n" %(semiaxes[i][0],semiaxes[i][1],semiaxes[i][2]))
		
		text_file.write("== masses %s \n" %mass_unit)
		for i in range(N_bodies):
			text_file.write("%.3e \n" %(masses[i]))
		
		if tolarr[6] == 0:
			text_file.write("== Positions (x, y, z) \n")
			for i in range(N_bodies):
				text_file.write("(%.2e, %.2e, %.2e) \n" %(y_init[N3+i*3], y_init[N3+i*3+1], y_init[N3+i*3+2]))
			
			text_file.write("== Velocities (vx, vy, vz) \n")
			for i in range(N_bodies):
				text_file.write("(%.2e, %.2e, %.2e) \n" %(y_init[i*3], y_init[i*3+1], y_init[i*3+2]))
		
		elif tolarr[6] == 1:
			text_file.write("== (period, eccentricity, inclination)  \n")
			for i in range(N_bodies):
				text_file.write("(%.2e, %.2e, %.2e) \n" %(y_init[N3+i*3], y_init[N3+i*3+1], y_init[N3+i*3+2]))
			
			text_file.write("== (ascending node, longitude of perihelion, mean longitude) \n")
			for i in range(N_bodies):
				text_file.write("(%.2e, %.2e, %.2e) \n" %(y_init[i*3], y_init[i*3+1], y_init[i*3+2]))
		
		elif tolarr[6] == 2:
			text_file.write("== (semimajor axis, eccentricity, inclination)  \n")
			for i in range(N_bodies):
				text_file.write("(%.2e, %.2e, %.2e) \n" %(y_init[N3+i*3], y_init[N3+i*3+1], y_init[N3+i*3+2]))
			
			text_file.write("== (ascending node, longitude of perihelion, mean anomaly) \n")
			for i in range(N_bodies):
				text_file.write("(%.2e, %.2e, %.2e) \n" %(y_init[i*3], y_init[i*3+1], y_init[i*3+2]))
				
		text_file.write("== Rotation angle (phi, theta, psi) \n")
		for i in range(N_bodies):
			text_file.write("(%.3f, %.3f, %.3f) \n" %(np.rad2deg(y_init[N9+i*3]), 
					np.rad2deg(y_init[N9+i*3+1]), np.rad2deg(y_init[N9+i*3+2])))			
		
		text_file.write("== Angular velocity (omega_x, omega_y, omega_z) \n")
		for i in range(N_bodies):
			text_file.write("(%.3f, %.3f, %.3f) \n" %(y_init[N6+i*3], y_init[N6+i*3+1], y_init[N6+i*3+2]))

def show_info_terminal(method, t_array, semiaxes, n_vertices, adaptive, usempir, unit_text):
	print("=======================")
	print("Using method: %s" %method)
	print("Units - %s/%s" %(unit_text[0], unit_text[1]))
	print("Time span: %.2f %s" %(t_array[-1], unit_text[1]))
	if adaptive:
		print("Adaptive stepper: ON")
	else:
		print("Adaptive stepper: OFF")
		print("Step size = %.2f" %(t_array[1]-t_array[0]))
	if usempir:
		print("mpir enabled")
	print("Body shapes:")
	for i in range(len(n_vertices)):
		if n_vertices[i] > 0:
			print("Body %d a polyhedron with %d vertices" %(i+1, n_vertices[i]))
		else:
			if semiaxes[i][0] > semiaxes[i][1] and semiaxes[i][1] > semiaxes[i][2]:
				print("Body %d an ellipsoid with semiaxes (%.2f, %.2f, %.2f)" 
					%(i+1, semiaxes[i][0], semiaxes[i][1], semiaxes[i][2]))
			elif np.abs(semiaxes[i][0] - semiaxes[i][1]) < 1e-15 and semiaxes[i][0] > semiaxes[i][2]:
				print("Body %d an oblate spheroid with semiaxes (%.2f, %.2f, %.2f)" 
					%(i+1, semiaxes[i][0], semiaxes[i][1], semiaxes[i][2]))
			elif np.abs(semiaxes[i][0] - semiaxes[i][1]) < 1e-15 and semiaxes[i][0] < semiaxes[i][2]:
				print("Body %d a prolate spheroid with semiaxes (%.2f, %.2f, %.2f)" 
					%(i+1, semiaxes[i][0], semiaxes[i][1], semiaxes[i][2]))
			else:
				print("Body %d a sphere with radius (%.2f)" %(i+1, semiaxes[i][0]))

	print("=======================")
				
def main(parsed_arguments, data_output_dir, saveEnergy=True, solextension=None):
	"""
	This is the main program.
	When this is called, the differential equations are solved.
	"""
	# Create folder to save the pickle files
	if not os.path.isdir(data_output_dir):
		os.makedirs(data_output_dir)

	if parsed_arguments.UseMpir:
		import cymodulesmpir.odetableaus
		import cymodulesmpir.other_functions_cy as OFcy
		import cymodulesmpir.ODESolvers as ODESolvers
	else:
		import cymodules.odetableaus
		import cymodules.other_functions_cy as OFcy
		import cymodules.ODESolvers as ODESolvers
	dnum = np.array([0])
	input_data = readinput.read_input_data(parsed_arguments.InputFile)
	t_array = input_data[0]
	masses = input_data[1]
	y_init = input_data[2]
	semiaxes = input_data[3]
	N_bodies = input_data[4]
	tolarr = input_data[5]
	mass_scale = input_data[6]
	binary_ids = input_data[7]
	vertices = input_data[8]
	face_ids = input_data[9]
	n_vertices = input_data[10]
	N_faces = input_data[11]
	vert_filenames = input_data[12]
	p_info = np.concatenate([n_vertices, N_faces]).astype(np.int32)
	G_g, time_convert, E_convert, unit_text, semiaxes = Set_G_parameter(semiaxes, parsed_arguments.useGgrav)
	# Scale the mass unit based on the largest mass of the bodies
	G_g *= mass_scale
	plot_save_folder = "PlotResults/" + parsed_arguments.FolderExtra + "/"
	results_dir = os.path.join(Folder_location, plot_save_folder)
	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)
	Textfile_out = data_output_dir + "InputParams.txt"
	if os.path.isfile(Textfile_out):
		os.remove(Textfile_out)
	write_out_param_data(Textfile_out, t_array, masses*mass_scale, y_init, semiaxes, N_bodies, tolarr,
						 unit_text, vert_filenames, parsed_arguments)
	if tolarr[7] == 1:
		inc_sun = True
	else:
		inc_sun = False
	if tolarr[6] == 1 or tolarr[6] == 2:
		if not parsed_arguments.useGgrav:
			raise ValueError("Dimensionless G_grav! Usage of Keplerian coordinates is not advised."
							 " \nSet -units to a unit if Keplerian elements are used.")
		y_init = Orbel_initial_conditions(y_init, N_bodies, masses, binary_ids, G_g, tolarr, time_convert, mass_scale)

	Use_euler_params = 0
	if parsed_arguments.UseEulerParams:
		y_init = TB_to_Euler_params_init(y_init, N_bodies)
		Use_euler_params = 1
	adaptive_stepper = 0
	if parsed_arguments.useAdaptive:
		adaptive_stepper = 1
	int_info = np.array([N_bodies, Use_euler_params, inc_sun, adaptive_stepper], dtype=np.int32)
	
	# Sets default tolerances in case it is set to zero in the input file
	default_tolerance_check(tolarr, parsed_arguments.ODESolver)
	start_time = time.time()
	if solextension:
		sol_pickle_filename = data_output_dir + 'ODE_solution%s.npz' %solextension
	else:
		sol_pickle_filename = data_output_dir + 'ODE_solution.npz'
	energies_pickle_filename = data_output_dir + "Energies.npz"
	scaled_mass = True
	
	if os.path.isfile(sol_pickle_filename):
		sol, semiaxes, masses = readsolution.read_solution(parsed_arguments.FolderExtra, 
									rundate=parsed_arguments.RunDate, simplify=False)
		G_g, time_convert, E_convert, unit_text, semiaxes = Set_G_parameter(semiaxes, parsed_arguments.useGgrav)
		mass_scale = 10**np.max(np.floor(np.log10(np.abs(masses))))
		masses /= mass_scale
		G_g *= mass_scale
	else:
		show_info_terminal(parsed_arguments.ODESolver, t_array, semiaxes, n_vertices, adaptive_stepper, 
						  parsed_arguments.UseMpir, unit_text)
		cysol = ODESolvers.ODESolvers(t_array, y_init, semiaxes, vertices, p_info, face_ids, masses, int_info, 
					tolarr, G_g, data_output_dir, method=parsed_arguments.ODESolver)
		cysol.sort_as_ysol()
		sol = CythonToScipy(cysol, N_bodies, Use_euler_params)
		if sol.NSolFiles == 1:
			readsolution.save_solution(data_output_dir, sol, semiaxes, masses*mass_scale, sol_pickle_filename, Use_euler_params)
	if saveEnergy:
		if os.path.isfile(energies_pickle_filename):
			U, Ek, Erot, Etot = readsolution.read_params(parsed_arguments.FolderExtra, rundate=parsed_arguments.RunDate)
		else:
			print("Computing energies")
			vertices, Body_volumes = OF.polyhedron_volume_centroid(vertices, semiaxes, p_info, face_ids)
			massdens = np.zeros(2*N_bodies+1)
			for i in range(N_bodies):
				if n_vertices[i] > 0:
					volume = Body_volumes[i]
				else:
					volume = (4.0/3.0)*np.pi*semiaxes[i][0]*semiaxes[i][1]*semiaxes[i][2]
				massdens[i] = masses[i]
				if volume == 0:
					massdens[i+N_bodies] = 0	
				else:
					massdens[i+N_bodies] = masses[i]/volume
			massdens[-1] = mass_scale
			
			if np.count_nonzero(n_vertices) > 2:
				print("Energy computation for more than 2 polyhedra not implemented.")
				U = np.zeros(len(sol.t))
				Ek = np.zeros(len(sol.t))
				Erot = np.zeros(len(sol.t))
				Etot = np.zeros(len(sol.t))
			else:
				if sol.NSolFiles > 1:
					for i in range(sol.NSolFiles):
						multisol = readsolution.ReadMultiSolutions(parsed_arguments.FolderExtra, i+1, N_bodies, 
									Use_euler_params, rundate=parsed_arguments.RunDate, simplify=False)
						int_info2 = np.array([len(multisol.t), N_bodies, Use_euler_params, inc_sun],dtype=np.int32)
						U, Ek, Erot, Etot = OFcy.Compute_energies_Nbody(semiaxes, massdens, multisol.var, int_info2, tolarr,
																			vertices, p_info, G_g, face_ids)
						energies_pickle_filename = data_output_dir + "EnergiesSol%d.npz" %(i+1)
						readsolution.save_params([U,Ek,Erot,Etot], energies_pickle_filename)
				else:
					int_info2 = np.array([len(sol.t), N_bodies, Use_euler_params, inc_sun],dtype=np.int32)
					U, Ek, Erot, Etot = OFcy.Compute_energies_Nbody(semiaxes, massdens, sol.var, int_info2, tolarr,
																		vertices, p_info, G_g, face_ids)
					readsolution.save_params([U,Ek,Erot,Etot], energies_pickle_filename)
	if parsed_arguments.Convertsolution:
		print("Writing to standard file format...")
		sol_filename = data_output_dir + 'ODE_solution.' + parsed_arguments.SolutionType
		energies_filename = data_output_dir + "Energies."  + parsed_arguments.SolutionType
		readsolution.save_basic_format(data_output_dir, sol, [U, Ek, Erot], sol_filename, energies_filename, Use_euler_params, N_bodies)

	print("Execution time --- %s seconds ---" % (time.time() - start_time))
	print("Number of evaluations = %d" %sol.nfev)
	
	if parsed_arguments.doplot:
		Energies = [U, Ek, Erot, Etot]
		Plot_results(sol, semiaxes, masses, Energies, N_bodies, mass_scale, unit_text, inc_sun, plot_save_folder)
	print("Numpy files saved in: ", data_output_dir)
	

def Plot_results(sol, semiaxes, masses, Energies, N_bodies, mass_scale, unit_text, include_sun, results_dir):
	""" Function that plots the results, including positions, orbits and energies. """
	import matplotlib.pyplot as plt
	# Create a folder that contains the output pdf files
	print("Figures saved in: ", results_dir)
	saveinstance = SaveFigures(results_dir, ".pdf")
	Rcm_x = 0
	Rcm_y = 0
	Rcm_z = 0
	for i in range(N_bodies):
		Rcm_x += (masses[i]*sol.var[N_bodies*3 + i*3])
		Rcm_y += (masses[i]*sol.var[N_bodies*3 + i*3 + 1])
		Rcm_z += (masses[i]*sol.var[N_bodies*3 + i*3 + 2])
	if include_sun:
		Rcm_x /= (np.sum(masses)+1.989*1e30)
		Rcm_y /= (np.sum(masses)+1.989*1e30)
		Rcm_z /= (np.sum(masses)+1.989*1e30)
	else:
		Rcm_x /= np.sum(masses)
		Rcm_y /= np.sum(masses)
		Rcm_z /= np.sum(masses)
	xcm = np.zeros((N_bodies, len(sol.t)))
	ycm = np.zeros((N_bodies, len(sol.t)))
	zcm = np.zeros((N_bodies, len(sol.t)))
	for i in range(N_bodies):
		xcm[i] = sol.var[N_bodies*3 + i*3] - Rcm_x
		ycm[i] = sol.var[N_bodies*3 + i*3 + 1] - Rcm_y
		zcm[i] = sol.var[N_bodies*3 + i*3 + 2] - Rcm_z

	# Plot 2D Orbit
	fig = plt.figure(figsize=(8,8))
	ax = plt.gca()
	xmax = np.max(xcm)
	ymax = np.max(ycm)
	xmin = np.min(xcm)
	ymin = np.min(ycm)
	xs = xmax - xmin
	ys = ymax - ymin
	ssize = max(xs,ys)
	xmid = (xmax + xmin)/2.
	ymid = (ymax + ymin)/2.
	
	plt.xlim(xmid-ssize/2.-2.,xmid+ssize/2.+2.)
	plt.ylim(ymid-ssize/2.-2.,ymid+ssize/2.+2.)
	for i in range(N_bodies):
		ctrl = N_bodies*3 + i*3
		plt.plot(xcm[i],ycm[i], label='Body %d' %i)
	plt.text(0.55,0.95,"t span = [%.1f, %.1f] %s" %(sol.t[0], sol.t[-1], unit_text[1]),fontsize=12,transform=ax.transAxes)
	plt.grid()
	plt.tight_layout()
	ax.set_aspect(1.0)
	if include_sun:
		plt.plot(0, 0, '*', label='Sun')
	plt.xlabel('x - [%s]' %unit_text[0])
	plt.ylabel('y - [%s]' %unit_text[0])
	plt.legend()
	saveinstance.savefigure(fig, "2Dorbits_NBody")

	U, Ek, Erot, Etot = Energies
	fig = plt.figure(figsize=(8,12))
	ax1 = fig.add_subplot(3,1,1)
	ax1.plot(sol.t,U,label='$U$', color='green')
	ax1.plot(sol.t,Ek,label='$E_k$', color='red')
	ax1.plot(sol.t,Erot,label='$E_{rot}$', color='orange')
	ax1.plot(sol.t,Etot,label='$E_{tot}$', color='blue')
	ax1.set_xticklabels([])
	if include_sun:
		ax1.set_ylabel(r'Energies - [J $M_\odot^{-1}$]')
	else:
		ax1.set_ylabel('Energies - [J$/10^{%.0f}]$' %(np.log10(mass_scale)))
	ax1.legend()
	ax2 = fig.add_subplot(3,1,2)
	ax2.plot(sol.t,Etot, label='$E_{tot}$', color='blue')
	ax2.legend()
	if include_sun:
		ax2.set_ylabel(r'Total energy - [J $M_\odot^{-1}$]')
	else:
		ax2.set_ylabel('Total energy - [J$/10^{%.0f}]$' %(np.log10(mass_scale)))
	ax2.set_xticklabels([])
	ax3 = fig.add_subplot(3,1,3)
	ax3.plot(sol.t[:-1],np.abs(np.diff(Etot)/Etot[:-1]),label=r'$\delta E_{tot}$', color='black')
	ax3.legend()
	ax3.set_xlabel('Time')
	ax3.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
	saveinstance.savefigure(fig, "Energies")
		
if __name__ == '__main__':
	# Parsing different optional arguments
	parsed_arguments = Argument_parser()
	if parsed_arguments.RunDate:
		runtime = parsed_arguments.RunDate
		rundate_var = parsed_arguments.RunDate
	else:
		rundate_var = runtime
	if parsed_arguments.FolderExtra:
		runtime = runtime +  "/" + parsed_arguments.FolderExtra + "/"
		plot_save_folder = plot_save_folder + "/" + parsed_arguments.FolderExtra + "/"
		
	data_output_dir = os.path.join(Folder_location, "OutputData/"+runtime)
	# If desired, deletes pickle files from a previous run (assuming it exist)
	# Useful to recompute everything from scratch
	if parsed_arguments.DeletePickle:
		if os.path.isdir(data_output_dir):
			print("!!WARNING: Will delete the save files from a previous run. \n" +
				  "You have 5 seconds to cancel if this is not what you intended")
			time.sleep(5)
			shutil.rmtree(data_output_dir, ignore_errors=True)  # Remove data folder if called
			print("Previous save files deleted. Recomputing everything...")
		else:
			print(data_output_dir, " does not exist. Nothing to delete!")
	main(parsed_arguments, data_output_dir)
