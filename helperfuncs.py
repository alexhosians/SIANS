import os
import time
import numpy as np
import shutil

class main_input():
	def __init__(self, inputfile="example", rundate=time.strftime("%Y/%m_%d/"),
				method="RK54", units=None, savemiddata=False, foldExtra=None,
				mpir=False, euler=False, adaptive=True, doplot=False,
				deleteOldFile=False, solconv=False, soltype="txt"):
		""" 
		Serves as an alternative to run the simulations without 
		specifying input commands in the command line.
		The simulations must be run from a separate python file.
		See documentation for example.

		For safety reasons, do not modify any parts of this file.
		"""
		if ".txt" in inputfile:
			self.InputFile = "initial_values/" + inputfile
		else:
			self.InputFile = "initial_values/" + inputfile + ".txt"
		if not foldExtra:
			runtime = rundate + inputfile + "/"
			self.FolderExtra = inputfile
		else:
			runtime = rundate + "/" + foldExtra + "/"
			self.FolderExtra = foldExtra

		self.valid_check(method, units, soltype)
		self.ODESolver = method
		self.useGgrav = units
		self.useAdaptive = adaptive
		self.UseMpir = mpir
		self.SaveMidData = savemiddata
		self.UseEulerParams = euler
		self.doplot = doplot
		self.RunDate = rundate
		self.Convertsolution = solconv
		self.SolutionType = soltype

		Folder_location = os.path.dirname(os.path.realpath(__file__))
		self.data_output_dir = os.path.join(Folder_location, "OutputData/"+runtime)
		if deleteOldFile:
			if os.path.isdir(self.data_output_dir):
				print("!!WARNING: Will delete pickle file from a previous run. \n" +
					  "You have 5 seconds to cancel if this is not what you intended")
				time.sleep(5)
				shutil.rmtree(self.data_output_dir, ignore_errors=True)  # Remove data folder if called
				print("Previous pickle data deleted. Recomputing everything...")
			else:
				print(self.data_output_dir, " does not exist. Nothing to delete!")
	
		

	def valid_check(self, method, units, soltype):
		""" 
		Used to check if input is correct.
		For safety reasons, do not remove this part of the class.
		"""
		method_checks = np.array(["RK4", "RK54", "V65", "V76", "V87", "V98", "F108", "F1210", "T98"])
		SI_checks = np.array(["SI", "kmhr", "kmd", "kmyr", "auyr", "aud", None])
		outputfile_checks = np.array(["txt", "csv"])
		# Do not remove these checks
		if not (method == method_checks).any():
			raise ValueError("Method %s is not valid. Try the following: \n"
				"-method RK4 \n"
				"-method RK54 \n"
				"-method V65 \n"
				"-method V76 \n"
				"-method V87 \n"
				"-method V98 \n"
				"-method T98 \n"
				"-method F108 \n"
				"-method F1210" %method)
		if not (units == SI_checks).any():
			raise ValueError("SI unit input %s not valid. Try the following: \n"
				"-units SI \n"
				"-units kmhr \n"
				"-units kmd \n"
				"-units kmyr \n"
				"-units aud \n"
				"-units auyr" %units)
		if not (soltype == outputfile_checks).any():
			raise ValueError("Outputfile %s not valid. Try the following: \n"
				"-soltype txt \n"
				"-soltype csv" %soltype)
		

def make_input_file(filename, masses, semiaxes, pos, vel, angle, angvel, BID=None,
					tend=200.0, Nstep=500, ntol=0, iabstol=1e-5, ireltol=1e-5,
					quadval=4, hmax=1.5, hmin=1e-5, kepler=0, sun=0, density=0,
					pmdist=0, floatacc="%.3f"):
	"""
	Creates a new input file for the software to read.
	This is only applicable for ellipsoid simulations.
	For custom polyhedron runs, manual edits are required.
	See exampleruns.py for example usage.

	Input arguments:
	- filename (string): Filename of the input file.
	- masses (1 x N array): The masses (or densities) of the N bodies.
	- semiaxes (N x 3 array): The semi-axes of the ellipsoids. 
							Sorted as [a, b, c] for each body, with a > b > c
							Units to be consistent with input unit usage.
	- pos (N x 3 array): The positions of the bodies.
						 Sorted as [x, y, z] for each body.
						 Units to be consistent with input unit usage.
	- vel (N x 3 array): The velocities of the bodies.
						 Sorted as [vx, vy, vz] for each body.
						 Units to be consistent with input unit usage.
	- angle (N x 3 array): The Tait-Bryan (or Euler) angles of the bodies.
						 Sorted as [phi, theta, psi] for each body.
						 phi, theta, psi corresponds to rotations about x, y and z axes.
						 Units in radians.
	- angvel (N x 3 array): The angular velocity of the bodies.
							Sorted as [omega_x, omega_y, omega_z] for each body.
							Corresponds to angular velocity in x, y, z axis.
	- BID (1 x N array): The body ID used to determine binary orbits.
						  Only required if kepler = 1

	Optional arguments (see documentation under section 3.1 for more details):
	- tend (float): End time of simulation
	- Nstep (int): Number of time steps of the simulation 
				   (not relevant for adaptive time steppers)
	- ntol (float): Error tolerance of the Runge-Kutta adaptive time stepper
	- iabstol (float): Absolute tolerance of the surface integrator error estimate
	- ireltol (float): Relative tolerance of the surface integrator error estimate
	- quadval (int): Integrator order for the QAG adaptive integrator
	- hmax (float): Maximum step size allowed for adaptive time stepper
	- hmin (float): Minimum step size allowed for adaptive time stepper
	- kepler (int): Determines which kepler variables are used for initial conditions (0 cartesian)
	- sun (int): Determines if the Sun is included as an external body
	- density (int): Determines if densities are used instead of masses. Density if set to 1.

	- floatacc (string): Number of digits included into input file
	"""
	if tend <= 0:
		raise ValueError("Time end cannot be smaller than 0")
	if Nstep <= 0:
		raise ValueError("Number of time steps cannot be smaller than 0")
	if quadval < 1 or quadval > 6:
		raise ValueError("Argument 'quadval' cannot be smaller than 1 or larger than 6")
	if sun != 1 and sun != 0:
		raise ValueError("Argument 'sun' must be equal to 0 or 1")
	if kepler != 0 and kepler != 1 and kepler != 2:
		raise ValueError("Argument 'kepler' must be equal to 0, 1 or 2")
	if density != 0 and density != 1:
		raise ValueError("Argument 'density' must be equal to 0 or 1")

	N_bodies = len(masses)
	if kepler > 0 and not BID:
		raise ValueError("Kepler > 0 and BID not included. Set BID array.")
	if kepler == 0:
		BID = np.zeros(N_bodies)

	if ".txt" in filename:
		filename = "initial_values/" + filename
	else:
		filename = "initial_values/" + filename + ".txt"
	
	floatpoint_accuracy = floatacc + "  " + floatacc + "  " + floatacc + "\n"
	with open(filename, 'w') as text_file:
		text_file.write("t_end: %.1f\n" %tend)
		text_file.write("N: %d\n" %Nstep)
		text_file.write("ntol: %.1e\n" %ntol)
		text_file.write("iabstol: %.1e\n" %iabstol)
		text_file.write("ireltol: %.1e\n" %ireltol)
		text_file.write("quadval: %d\n" %quadval)
		text_file.write("hmax: %.1f\n" %hmax)
		text_file.write("hmin: %.1e\n" %hmin)
		text_file.write("kepler: %d\n" %kepler)
		text_file.write("sun: %d\n" %sun)
		text_file.write("density: %d\n\n" %density)

		text_file.write("@ Masses (or densities. Use density: 1 above): \n")
		for i in range(N_bodies):
			text_file.write("%.2e\n" %masses[i])
		text_file.write("\n")

		text_file.write("@ Semiaxes (a,b,c): \n")
		for i in range(N_bodies):
			text_file.write(floatpoint_accuracy %(semiaxes[i][0], semiaxes[i][1], semiaxes[i][2]))
		text_file.write("\n")
		
		if kepler == 1:
			text_file.write("@ Ascending node, Longitude of perihelion and Mean longitude\n")
		elif kepler == 2:
			text_file.write("@ Ascending node, Longitude of perihelion and Mean anomaly\n")
		else:
			text_file.write("@ Positions (x,y,z): \n")
		for i in range(N_bodies):
			text_file.write(floatpoint_accuracy %(pos[i][0], pos[i][1], pos[i][2]))
		text_file.write("\n")
		
		if kepler == 1:
			text_file.write("@ Period, Eccentricity and Inclination\n")
		elif kepler == 2:
			text_file.write("@ Semi-major axis, Eccentricity and Inclination\n")	
		else:
			text_file.write("@ Velocities (vx, vy, vz): \n")
		for i in range(N_bodies):
			text_file.write(floatpoint_accuracy %(vel[i][0], vel[i][1], vel[i][2]))
		text_file.write("\n")
				
		text_file.write("@ Angles (phi, theta, psi): \n")
		for i in range(N_bodies):
			text_file.write(floatpoint_accuracy %(angle[i][0], angle[i][1], angle[i][2]))
		text_file.write("\n")
		
		text_file.write("@ Angular velocity (omega_x, omega_y, omega_z): \n")
		for i in range(N_bodies):
			text_file.write(floatpoint_accuracy %(angvel[i][0], angvel[i][1], angvel[i][2]))
		text_file.write("\n")
		
		text_file.write("@ BID (only required if kepler > 0): \n")
		for i in range(N_bodies):
			text_file.write("%d\n" %(BID[i]))
		
		
		
		
		
		
		
		