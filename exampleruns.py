"""
This file shows some examples on how to run the software
without calling the code from the command line.

See the comments on each function for more details.

A separate python file may also be created to run the simulations.
Make sure to include the two import calls below.
"""
from main import main                    # This is required
from helperfuncs import main_input       # This is required
from helperfuncs import make_input_file  # Optional function

def simple_run():
	""" 
	A simple two-body run. 
	Simulation calls for the example.txt file.
	Saves data to a folder called "test"
	Uses order 9(8) Runge-Kutta method by Verner
	Also allows the software to save some basic plots
	
	If this code segment is rerun, deletes the old results.
	To prevent deletion of old results, set deleteOldFile=False
	"""
	arginput = main_input(method="V98", foldExtra="test", doplot=True, deleteOldFile=True)
	### If one desires the results in a readable format, onecanuse the 'solconv' argument
	### Remove the comment in the line below, and add a comment # in the executing line above
	#arginput = main_input(method="V98", foldExtra="test", doplot=True, deleteOldFile=True, solconv=True)
	
	main(arginput, arginput.data_output_dir)

def binary1999KW4():
	""" 
	Simulations the asteroid binary 1999KW4 (Moshup and Squannit).
	This simulation reproduces the results of Ho et al. (2021).
	See: https://ui.adsabs.harvard.edu/abs/2021CeMDA.133...35H/abstract
	To obtain the spheroid results, switch semiaxes so that a = b in the 1999KW input file.
	See the input file for more details.
	"""
	arginput = main_input(inputfile="1999KW4", method="V98", foldExtra="1999KW4", units="kmhr", doplot=True)
	main(arginput, arginput.data_output_dir)

def three_body_test():
	""" 
	A simple 3 body test simulation.
	Uses the Dormand-Prince order 5(4) method.
	"""
	arginput = main_input(inputfile="3bodytest", doplot=True, deleteOldFile=True)
	main(arginput, arginput.data_output_dir)

def create_new_input():
	"""
	Instead of creating a new input file manually, it can be created through a helper function.
	This is useful if many simulations are to be run.
	Creates a simple two-body problem with ellipsoids.

	Uses quadval = 5 for integrator
	"""
	import numpy as np
	inputfile = "NewExample"
	masses = np.array([5.0, 5.0])
	
	semiaxes = np.array([[1.0, 0.7, 0.6], 
						 [1.0, 0.9, 0.55]])
	
	position = np.array([[5.0, 0.0, 1.0],
						 [-6.0, 2.0, 0.0]])

	velocity = np.array([[0.0, 0.4, 0.0],
						 [0.0, -0.4, 0.0]])

	angle = np.array([[0.0, np.pi/7, 0.0],
					  [np.pi/13, 0.0, 0.0]])

	angularvel = np.array([[0.0, 0.0, 1e-3],
					   	   [0.0, 0.0, 1e-3]])
	make_input_file(inputfile, masses, semiaxes, position, velocity, angle, angularvel, 
					quadval=5, floatacc="%.6f", Nstep=100)
	
	arginput = main_input(inputfile=inputfile, doplot=True, adaptive=False)
	main(arginput, arginput.data_output_dir)

def obtain_force_U():
	"""
	A simple example on how to only grab the forces or torques from the surface integration method at one instance.
	Applicable for bodies of arbitrary shapes.
	Polyhedron shapes must have triangular faces.

	The force computing function is in 'other_functions_cy.pyx' files, found in cymodules or cymodulesmpir
	Whichever is chosen does not matter. 
	However, the latter option is safer to avoid round-off errors in the polyhedron potential.

	Body A is being integrated over, while in the gravity field of body B
	"""
	import numpy as np
	import cymodules.other_functions_cy as OFcy
	# Or if mpir is enabled 
	#import cymodulesmpir.other_functions_cy as OFcy

	# Gravitational constant
	G_grav = 6.67408e-11 
	#######
	####### Two ellipsoid example
	#######
	# Positions of the bodies - Units of meters
	x_A = -1000.0
	y_A = 0.0
	z_A = 0.0
	x_B = 1000.0
	y_B = 0.0
	z_B = 0.0
	position = np.array([x_A, y_A, z_A, x_B, y_B, z_B])
	# Rotation angles - Follows Tait-Bryan convention, see documentation
	# Units in radians
	phi_A = 0.0
	theta_A = 0.0
	psi_A = 0.0
	phi_B = 0.0
	theta_B = 0.0
	psi_B = 0.0
	angles = np.array([phi_A, theta_A, psi_A, phi_B, theta_B, psi_B])
	# Densities - Units of kg/m^3
	rho_A = 2000.0
	rho_B = 2000.0
	# Semiaxes of the ellipsoids - Units of meters
	a_A = 1000.0
	b_A = 750.0 
	c_A = 600.0
	a_B = 700.0
	b_B = 500.0 
	c_B = 300.0
	semiaxes = np.array([a_A, b_A, c_A, a_B, b_B, c_B])
	m_A = 4*np.pi*rho_A*a_A*b_A*c_A/3
	m_B = 4*np.pi*rho_B*a_B*b_B*c_B/3
	massdens = np.array([m_A, m_B, rho_A, rho_B])
	# Required in case of polyhedron shapes
	nfaces = np.array([0, 0])  # Number of faces
	nverts = np.array([0, 0])  # Number of vertices
	vertices = np.array([])    # No vertices, empty array
	face_ids = np.array([], dtype=np.int32)    # No faces, empty array
	# Choose between obtaining forces or torques
	# FM_check = 1 -> Forces
	# FM_check = 2 -> Torques
	FM_check = 1
	### Beware of round-off errors in the surface integration scheme
	### May be useful to round off small values
	### If roundoff >= 1, rounds all values smaller than 10^(-16) to 0
	roundoff = 0
	Fx, Fy, Fz = OFcy.getforce(position, angles, massdens, semiaxes, nfaces, nverts, vertices, face_ids, FM_check, G_grav, roundoff=roundoff)
	print(Fx, Fy, Fz)
	### Example on obtaining potential energy. Uses the same parameters.
	U = OFcy.getpotential(position, angles, massdens, semiaxes, nfaces, nverts, vertices, face_ids, G_grav)
	print(U)
	
	#######
	####### Ellipsoid - Tetrahedron example
	#######
	## Dimensionless quantities
	G_grav = 1.0
	# Tetrahedron shape same as 'tetrahedron.obj' file in the 'polyhedrondata' folder
	# Positions of the bodies - Dimensionless
	x_A = -4.0
	y_A = 0.0
	z_A = 0.0
	x_B = 4.0
	y_B = 0.0
	z_B = 0.0
	position = np.array([x_A, y_A, z_A, x_B, y_B, z_B])
	# Rotation angles - Follows Tait-Bryan convention, see documentation
	# Units in radians
	phi_A = 0.0
	theta_A = 0.0
	psi_A = 0.0
	phi_B = 0.0
	theta_B = 0.0
	psi_B = 0.0
	angles = np.array([phi_A, theta_A, psi_A, phi_B, theta_B, psi_B])
	# Densities - Dimensionless
	rho_A = 2.0
	rho_B = 2.0
	# Semiaxes of the ellipsoids - Dimensionless
	a_A = 1.0
	b_A = 0.7 
	c_A = 0.5
	a_B = 1.0 # For tetrahedron, these do not matter as they are ignored
	b_B = 0.7 
	c_B = 0.5
	semiaxes = np.array([a_A, b_A, c_A, a_B, b_B, c_B])
	# Masses of polyhedron is determined through a different algorithm. See Dobrovolskis (1996)
	m_A = 4*np.pi*rho_A*a_A*b_A*c_A/3
	m_tet = 2.0/3
	massdens = np.array([m_A, m_tet, rho_A, rho_B])
	nfaces = np.array([0, 4], dtype=np.int32)  # Number of faces
	nverts = np.array([0, 4], dtype=np.int32)  # Number of vertices
	vertices = np.array([0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5])
	face_ids = np.array([2, 1, 0, 0, 1, 3, 0, 2, 3, 1, 2, 3], dtype=np.int32)
	Fx, Fy, Fz = OFcy.getforce(position, angles, massdens, semiaxes, nfaces, nverts, vertices, face_ids, FM_check, G_grav, roundoff=0)
	print(Fx, Fy, Fz)
	U = OFcy.getpotential(position, angles, massdens, semiaxes, nfaces, nverts, vertices, face_ids, G_grav)
	print(U)
	
	#######
	####### Two tetrahedra example
	#######
	# Both tetrahedra shapes are the same as 'tetrahedron.obj' file in the 'polyhedrondata' folder
	# Paramaters are identical to the ellipsoid - tetrahedron case
	# Only change is the vertices
	massdens = np.array([m_tet, m_tet, rho_A, rho_B])
	nfaces = np.array([4, 4], dtype=np.int32)  # Number of faces
	nverts = np.array([4, 4], dtype=np.int32)  # Number of vertices
	vertices = np.array([0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5,   # Tetrahedron 1
						 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5])  # Tetrahedron 2
	face_ids = np.array([2, 1, 0, 0, 1, 3, 0, 2, 3, 1, 2, 3, # Tetrahedron 1
						 2, 1, 0, 0, 1, 3, 0, 2, 3, 1, 2, 3], dtype=np.int32) # Tetrahedron 2
	Fx, Fy, Fz = OFcy.getforce(position, angles, massdens, semiaxes, nfaces, nverts, vertices, face_ids, FM_check, 1.0, roundoff=0)
	print(Fx, Fy, Fz)
	U = OFcy.getpotential(position, angles, massdens, semiaxes, nfaces, nverts, vertices, face_ids, G_grav)
	print(U)
	#######
	####### Tetrahedron - Octahedron example
	#######
	# Shapes are given by the respective .obj files in the 'polyhedrondata' folder
	# Paramaters are identical to the ellipsoid - tetrahedron case
	# Only change is the vertices
	m_oct = 8.0/3.0
	massdens = np.array([m_tet, m_oct, rho_A, rho_B])
	nfaces = np.array([4, 8], dtype=np.int32)  # Number of faces
	nverts = np.array([4, 6], dtype=np.int32)  # Number of vertices
	vertices = np.array([0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5,   # Tetrahedron
						 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1])  # Octahedron
	face_ids = np.array([2, 1, 0, 0, 1, 3, 0, 2, 3, 1, 2, 3, # Tetrahedron
						 4, 2, 0, 0, 2, 5, 0, 3, 4, 0, 3, 5, 1, 2, 4, 1, 2, 5, 4, 3, 1, 1, 3, 5], dtype=np.int32) # Octahedron
	Fx, Fy, Fz = OFcy.getforce(position, angles, massdens, semiaxes, nfaces, nverts, vertices, face_ids, FM_check, 1.0, roundoff=0)
	print(Fx, Fy, Fz)
	U = OFcy.getpotential(position, angles, massdens, semiaxes, nfaces, nverts, vertices, face_ids, G_grav)
	print(U)
	

	
if __name__ == '__main__':
	# For testing purposes
	# Comment out different function calls for examples
	simple_run()
	#binary1999KW4()
	#three_body_test()
	#create_new_input()
	#obtain_force_U()