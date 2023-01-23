# cython: linetrace=True
# Import C square root call
import numpy as np
import cython
from libc.stdlib cimport malloc, free
from libc.math cimport pi, cos, sin, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Compute_energies_tetrahedron_2body(double[:,:] semiaxes, double[:] massdens, double[:,:] sol, int[:] int_info,
 double[:] tolerance_arrs, double[:] vertices1D, int[:] p_info, double G_grav, int[:] index_ordering):
	"""
	Computes the energy of the system.
	Intended to work for two-body case of arbitrary shapes.
	"""

	cdef Py_ssize_t i, j, k, l
	cdef int N_step = int_info[0]
	cdef int N_Bods = int_info[1]
	cdef int eulerparam = int_info[2]
	cdef int include_sun = int_info[3]

	cdef double a_A = semiaxes[0][0]
	cdef double b_A = semiaxes[0][1]
	cdef double c_A = semiaxes[0][2]
	cdef double a_B = semiaxes[1][0]
	cdef double b_B = semiaxes[1][1]
	cdef double c_B = semiaxes[1][2]
	cdef double mA = massdens[0]
	cdef double mB = massdens[1]
	cdef double rho_A = massdens[2]
	cdef double rho_B = massdens[3]
	cdef double mass_scale = massdens[4]
	
	cdef double[:] U = np.zeros(N_step, dtype=np.float64)
	cdef double[:] Ek = np.zeros(N_step, dtype=np.float64)
	cdef double[:] Erot = np.zeros(N_step, dtype=np.float64)
	cdef double[:] Etot = np.zeros(N_step, dtype=np.float64)

	cdef double[:,:] vx = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] vy = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] vz = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] x = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] y = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] z = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] omega_x = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] omega_y = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] omega_z = np.zeros((N_Bods, N_step), dtype=np.float64)

	cdef double[:,:] phi, theta, psi
	cdef double[:,:] e0, e1, e2, e3
	if eulerparam:
		e0 = np.zeros((N_Bods, N_step), dtype=np.float64)
		e1 = np.zeros((N_Bods, N_step), dtype=np.float64)
		e2 = np.zeros((N_Bods, N_step), dtype=np.float64)
		e3 = np.zeros((N_Bods, N_step), dtype=np.float64)
	else:
		phi = np.zeros((N_Bods, N_step), dtype=np.float64)
		theta = np.zeros((N_Bods, N_step), dtype=np.float64)
		psi = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef int iC, iD
	for i in range(N_Bods):
		iC = i*3
		iD = i*4
		vx[i] = sol[iC]
		vy[i] = sol[iC+1]
		vz[i] = sol[iC+2]
		x[i] = sol[N_Bods*3 + iC]
		y[i] = sol[N_Bods*3 + iC + 1]
		z[i] = sol[N_Bods*3 + iC + 2]
		omega_x[i] = sol[N_Bods*6 + iC]
		omega_y[i] = sol[N_Bods*6 + iC + 1]
		omega_z[i] = sol[N_Bods*6 + iC + 2]
		if eulerparam:
			e0[i] = sol[N_Bods*9 + iD + 0]
			e1[i] = sol[N_Bods*9 + iD + 1]
			e2[i] = sol[N_Bods*9 + iD + 2]
			e3[i] = sol[N_Bods*9 + iD + 3]
		else:
			phi[i] = sol[N_Bods*9 + iC]
			theta[i] = sol[N_Bods*9 + iC + 1]
			psi[i] = sol[N_Bods*9 + iC + 2]

	cdef double xc, yc, zc
	cdef double E_rotA, E_rotB, Ep, Eptemp
	cdef double two_pi = 2.0*pi
	cdef double half = (1.0/2.0)

	cdef double[4] limits = [-c_A, c_A, 0.0, two_pi]
	cdef double[6] saxes
	cdef double[4] massdens_in
	cdef int vertexids[6]
	cdef int vertex_combo[3]
	cdef double input_values[11]
	cdef double[3] itols = [tolerance_arrs[1], tolerance_arrs[2], tolerance_arrs[3]]

	cdef double Ea1, Ea2, Ea3
	cdef double vA, vB
	cdef double[:,:] I = np.zeros((3,3), dtype=np.float64)
	cdef double[:] I_1D
	cdef double[:] vertices_tetra_in
	cdef int[:] face_ids_memview
	
	cdef int id_fac, total_faces
	cdef int total_vertices = 0
	for i in range(N_Bods):
		total_vertices += p_info[i]
	cdef int* face_ids = <int *>malloc(3*p_info[3]*sizeof(int))
	cdef double* vertices = <double *>malloc(3*total_vertices*sizeof(double))
	for i in range(3*total_vertices):
		vertices[i] = vertices1D[i]
	
	for i in range(N_step):
		Ep = 0.0
		xc = x[1][i] - x[0][i]
		yc = y[1][i] - y[0][i]
		zc = z[1][i] - z[0][i]
		if eulerparam:
			input_values[:] = [xc, yc, zc, e0[0][i], e1[0][i], e2[0][i], e3[0][i], e0[1][i], e1[1][i], e2[1][i], e3[1][i]]
		else:
			input_values[:] = [xc, yc, zc, phi[0][i], theta[0][i], psi[0][i], phi[1][i], theta[1][i], psi[1][i], 0, 0] 
		vertexids[:] = [p_info[0], p_info[1], p_info[2], p_info[3], 0, 1]
		massdens_in[:] = [rho_A, rho_B, mB, G_grav]
		saxes[:] = [a_A, b_A, c_A, a_B, b_B, c_B]
		for j in range(p_info[3]):
			face_ids[3*j] = index_ordering[3*j + 3*p_info[2]]
			face_ids[3*j + 1] = index_ordering[3*j + 3*p_info[2] + 1]
			face_ids[3*j + 2] = index_ordering[3*j + 3*p_info[2] + 2]
		if p_info[2] > 0:
			for j in range(p_info[2]):
				vertex_combo[:] = [index_ordering[3*j], index_ordering[3*j+1], index_ordering[3*j+2]]
				Ep += mutual_potential_polyhedron(saxes, input_values, itols, vertices,
														vertex_combo, massdens_in, face_ids, vertexids, eulerparam)
		else:
			if a_A == b_A and a_A == c_A:
				#Ep = G_grav*mA*mB/sqrt(xc*xc + yc*yc + zc*zc)
				massdens_in[:] = [mA, mB, rho_B, G_grav]
				Ep += mutual_potential_point_mass(saxes, input_values, vertices, massdens_in, face_ids, vertexids, eulerparam)
			else:
				Ep += mutual_potential_ellipsoid(limits, saxes, input_values, itols, vertices, vertexids,
														 	face_ids, massdens_in, eulerparam)
		
		# Potential energy from the Sun
		#
		# NOTE:
		# All energy scalings are divided by sun mass.
		# Multiply with sun mass again to gain SI units.
		#
		if include_sun == 1:
			massdens_in[:] = [rho_A, 1, 1, G_grav]
			saxes[:] = [a_A, b_A, c_A, 0, 0, 0]
			if eulerparam:
				input_values[:] = [x[0][i], y[0][i], z[0][i], e0[0][i], e1[0][i], e2[0][i], e3[0][i], 0, 0, 0, 0]
			else:
				input_values[:] = [x[0][i], y[0][i], z[0][i], phi[0][i], theta[0][i], psi[0][i], 0, 0, 0, 0, 0] 
			if p_info[2] > 0:
				for j in range(p_info[2]):
					vertex_combo[:] = [index_ordering[3*j], index_ordering[3*j+1], index_ordering[3*j+2]]
					Ep += mutual_potential_polyhedron(saxes, input_values, itols, vertices,
														vertex_combo, massdens_in, face_ids, vertexids, eulerparam)
			else:
				if a_A == 0:
					Ep += G_grav*mA/sqrt(x[0][i]**2 + y[0][i]**2 + z[0][i]**2)
				else:
					Ep += mutual_potential_ellipsoid(limits, saxes, input_values, itols, vertices, vertexids,
															 	face_ids, massdens_in, eulerparam)
		id_fac = 0
		total_faces = 0
		for j in range(2):
			if p_info[j] > 0:
				face_ids_memview = np.zeros(3*p_info[N_Bods+j], dtype=np.int32)
				for k in range(p_info[N_Bods+j]):
					face_ids_memview[3*k] = index_ordering[3*k + total_faces]
					face_ids_memview[3*k + 1] = index_ordering[3*k + total_faces + 1]
					face_ids_memview[3*k + 2] = index_ordering[3*k + total_faces + 2]
				I_1D = moment_inertia_polyhedron(vertices, face_ids_memview, p_info[N_Bods+j], p_info[j], massdens[j], id_fac)
				for k in range(3):
					for l in range(3):
						I[k][l] = I_1D[3*k + l]
				id_fac += 3*p_info[j]
				total_faces += 3*p_info[N_Bods+j]
			else:
				moment_inertia_ellipsoid_matrix(semiaxes[j][0], semiaxes[j][1], semiaxes[j][2], massdens[N_Bods+j], I)
			Ea1 = omega_x[j][i]*(omega_x[j][i]*I[0][0] + omega_y[j][i]*I[0][1] + omega_z[j][i]*I[0][2])
			Ea2 = omega_y[j][i]*(omega_x[j][i]*I[1][0] + omega_y[j][i]*I[1][1] + omega_z[j][i]*I[1][2])
			Ea3 = omega_z[j][i]*(omega_x[j][i]*I[2][0] + omega_y[j][i]*I[2][1] + omega_z[j][i]*I[2][2])
			Erot[i] += half*(Ea1 + Ea2 + Ea3)*mass_scale
		vA = vx[0][i]**2 + vy[0][i]**2 + vz[0][i]**2
		vB = vx[1][i]**2 + vy[1][i]**2 + vz[1][i]**2
		U[i] = -Ep*mass_scale
		Ek[i] = half*(mA*vA + mB*vB)*mass_scale
		Etot[i] = Ek[i] + U[i] + Erot[i]
		
	free(vertices)
	free(face_ids)
	return U.base, Ek.base, Erot.base, Etot.base

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Compute_energies_Nbody(double[:,:] semiaxes, double[:] massdens, double[:,:] sol, int[:] int_info,
 double[:] tolerance_arrs, double[:] vertices1D, int[:] p_info, double G_grav, int[:] index_ordering):
	"""
	Computes the energy of the system.
	Generalized to N-body systems for bodies of arbitrary shape.
	"""

	cdef Py_ssize_t i, j, k, m
	cdef int N_step = int_info[0]
	cdef int N_Bods = int_info[1]
	cdef int eulerparam = int_info[2]
	cdef int include_sun = int_info[3]

	cdef double a_A, b_A, c_A 
	cdef double a_B, b_B, c_B 
	cdef double mA, mB, rho_A, rho_B
	cdef double mass_scale = massdens[2*N_Bods]
	
	cdef double[:] U = np.zeros(N_step, dtype=np.float64)
	cdef double[:] Ek = np.zeros(N_step, dtype=np.float64)
	cdef double[:] Erot = np.zeros(N_step, dtype=np.float64)
	cdef double[:] Etot = np.zeros(N_step, dtype=np.float64)

	cdef double[:,:] vx = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] vy = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] vz = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] x = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] y = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] z = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] omega_x = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] omega_y = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef double[:,:] omega_z = np.zeros((N_Bods, N_step), dtype=np.float64)

	cdef double[:,:] phi, theta, psi
	cdef double[:,:] e0, e1, e2, e3
	if eulerparam:
		e0 = np.zeros((N_Bods, N_step), dtype=np.float64)
		e1 = np.zeros((N_Bods, N_step), dtype=np.float64)
		e2 = np.zeros((N_Bods, N_step), dtype=np.float64)
		e3 = np.zeros((N_Bods, N_step), dtype=np.float64)
	else:
		phi = np.zeros((N_Bods, N_step), dtype=np.float64)
		theta = np.zeros((N_Bods, N_step), dtype=np.float64)
		psi = np.zeros((N_Bods, N_step), dtype=np.float64)
	cdef int iC, iD
	for i in range(N_Bods):
		iC = i*3
		iD = i*4
		vx[i] = sol[iC]
		vy[i] = sol[iC+1]
		vz[i] = sol[iC+2]
		x[i] = sol[N_Bods*3 + iC]
		y[i] = sol[N_Bods*3 + iC + 1]
		z[i] = sol[N_Bods*3 + iC + 2]
		omega_x[i] = sol[N_Bods*6 + iC]
		omega_y[i] = sol[N_Bods*6 + iC + 1]
		omega_z[i] = sol[N_Bods*6 + iC + 2]
		if eulerparam:
			e0[i] = sol[N_Bods*9 + iD + 0]
			e1[i] = sol[N_Bods*9 + iD + 1]
			e2[i] = sol[N_Bods*9 + iD + 2]
			e3[i] = sol[N_Bods*9 + iD + 3]
		else:
			phi[i] = sol[N_Bods*9 + iC]
			theta[i] = sol[N_Bods*9 + iC + 1]
			psi[i] = sol[N_Bods*9 + iC + 2]

	cdef double xc, yc, zc
	cdef double E_rotA, E_rotB, Ep, Eptemp
	cdef double two_pi = 2.0*pi

	cdef double[4] limits
	cdef double[6] saxes
	cdef double[4] massdens_in
	cdef int vertexids[6]
	cdef int vertex_combo[3]
	cdef double input_values[11]
	cdef double[3] itols = [tolerance_arrs[1], tolerance_arrs[2], tolerance_arrs[3]]

	cdef double Ea1, Ea2, Ea3
	cdef double vA, vB
	cdef double[:,:,:] I = np.zeros((N_Bods,3,3), dtype=np.float64)
	cdef double[:] I_1D
	cdef double[:] vertices_tetra_in
	cdef int[:] face_ids_memview
	
	cdef int total_vertices = 0
	for i in range(N_Bods):
		total_vertices += p_info[i]
	cdef int* face_ids = <int *>malloc(3*p_info[3]*sizeof(int))
	cdef double* vertices = <double *>malloc(3*total_vertices*sizeof(double))
	for i in range(3*total_vertices):
		vertices[i] = vertices1D[i]

	# Fill moments of inertia
	cdef int id_fac = 0
	cdef int total_faces = 0
	for j in range(N_Bods):
		if p_info[N_Bods+j] > 0:
			face_ids_memview = np.zeros(3*p_info[N_Bods+j], dtype=np.int32)
			for k in range(p_info[N_Bods+j]):
				face_ids_memview[3*k] = index_ordering[3*k + total_faces]
				face_ids_memview[3*k + 1] = index_ordering[3*k + total_faces + 1]
				face_ids_memview[3*k + 2] = index_ordering[3*k + total_faces + 2]
			I_1D = moment_inertia_polyhedron(vertices, face_ids_memview, p_info[N_Bods+j], p_info[j], massdens[j], id_fac)
			for k in range(3):
				for m in range(3):
					I[j][k][m] = I_1D[3*k + m]
			id_fac += 3*p_info[j]
			total_faces += 3*p_info[N_Bods+j]
		else:
			moment_inertia_ellipsoid_matrix(semiaxes[j][0], semiaxes[j][1], semiaxes[j][2], massdens[N_Bods+j], I[j])
			
	for i in range(N_step):
		Ep = 0
		for j in range(N_Bods):
			a_A = semiaxes[j][0]
			b_A = semiaxes[j][1]
			c_A = semiaxes[j][2]
			mA = massdens[j]
			rho_A = massdens[N_Bods+j]
			# Compute kinetic energy first
			Ea1 = omega_x[j][i]*(omega_x[j][i]*I[j][0][0] + omega_y[j][i]*I[j][0][1] + omega_z[j][i]*I[j][0][2])
			Ea2 = omega_y[j][i]*(omega_x[j][i]*I[j][1][0] + omega_y[j][i]*I[j][1][1] + omega_z[j][i]*I[j][1][2])
			Ea3 = omega_z[j][i]*(omega_x[j][i]*I[j][2][0] + omega_y[j][i]*I[j][2][1] + omega_z[j][i]*I[j][2][2])
			Erot[i] += 0.5*(Ea1 + Ea2 + Ea3)*mass_scale
			vA = vx[j][i]**2 + vy[j][i]**2 + vz[j][i]**2
			Ek[i] += 0.5*(mA*vA)*mass_scale
			# Potential energy
			for k in range(j+1, N_Bods):
				a_B = semiaxes[k][0]
				b_B = semiaxes[k][1]
				c_B = semiaxes[k][2]
				mB = massdens[k]
				rho_B = massdens[N_Bods+k]
				
				xc = x[k][i] - x[j][i]
				yc = y[k][i] - y[j][i]
				zc = z[k][i] - z[j][i]
				if eulerparam:
					input_values[:] = [xc, yc, zc, e0[j][i], e1[j][i], e2[j][i], e3[j][i], e0[k][i], e1[k][i], e2[k][i], e3[k][i]]
				else:
					input_values[:] = [xc, yc, zc, phi[j][i], theta[j][i], psi[j][i], phi[k][i], theta[k][i], psi[k][i], 0, 0] 
				vertexids[:] = [p_info[j], p_info[k], p_info[N_Bods+j], p_info[N_Bods+k], j, k]
				massdens_in[:] = [rho_A, rho_B, mB, G_grav]
				saxes[:] = [a_A, b_A, c_A, a_B, b_B, c_B]
				for m in range(p_info[N_Bods+k]):
					face_ids[3*m] = index_ordering[3*m + 3*p_info[2]]
					face_ids[3*m + 1] = index_ordering[3*m + 3*p_info[2] + 1]
					face_ids[3*m + 2] = index_ordering[3*m + 3*p_info[2] + 2]
				if p_info[N_Bods+j] > 0:
					# For polyhedron
					for m in range(p_info[N_Bods+j]):
						vertex_combo[:] = [index_ordering[3*m], index_ordering[3*m+1], index_ordering[3*m+2]]
						Ep += mutual_potential_polyhedron(saxes, input_values, itols, vertices,
															vertex_combo, massdens_in, face_ids, vertexids, eulerparam)
				else:
					if (a_A == b_A and a_A == c_A) or a_A == 0:
						# For sphere
						Ep += mutual_potential_point_mass(saxes, input_values, vertices, massdens_in, face_ids, vertexids, eulerparam)
					else:
						# For ellipsoid
						limits[:] = [-c_A, c_A, 0.0, two_pi]
						Ep += mutual_potential_ellipsoid(limits, saxes, input_values, itols, vertices, vertexids,
																 	face_ids, massdens_in, eulerparam)
			# Potential energy from the Sun, if applicable
			if include_sun == 1:
				massdens_in[:] = [rho_A, 1, 1, G_grav]
				saxes[:] = [a_A, b_A, c_A, 0, 0, 0]
				if eulerparam:
					input_values[:] = [x[j][i], y[j][i], z[j][i], e0[j][i], e1[j][i], e2[j][i], e3[j][i], 0, 0, 0, 0]
				else:
					input_values[:] = [x[j][i], y[j][i], z[j][i], phi[j][i], theta[j][i], psi[j][i], 0, 0, 0, 0, 0] 
				
				if p_info[N_Bods+j] > 0:
					for k in range(p_info[N_Bods+j]):
						vertex_combo[:] = [index_ordering[3*m], index_ordering[3*m+1], index_ordering[3*m+2]]
						Ep += mutual_potential_polyhedron(saxes, input_values, itols, vertices,
																	vertex_combo, massdens_in, face_ids, vertexids, eulerparam)
				else:
					if (a_A == b_A and a_A == c_A) or a_A == 0:
						Ep += G_grav*mA/sqrt(x[j][i]**2 + y[j][i]**2 + z[j][i]**2)
					else:
						limits[:] = [-c_A, c_A, 0.0, two_pi]
						Ep += mutual_potential_ellipsoid(limits, saxes, input_values, itols, vertices, vertexids,
																	 	face_ids, massdens_in, eulerparam)
		U[i] = -Ep*mass_scale
		Etot[i] = Ek[i] + U[i] + Erot[i]
	free(vertices)
	free(face_ids)
	return U.base, Ek.base, Erot.base, Etot.base

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef polyhedron_volume(double[:] vertices, int nvertices, int num_faces, int[:] face_ids):
	"""
	Computes the volume of the polyhedron and its centroid.
	Moves the vertices so they are centered around the centroid.

	Algorithm as follows:
	1) Consider a face that is a triangle. Create a tetrahedron with a fourth point that is the centroid.
	2) Compute the volume of said tetrahedron.
	3) Sum the volume of all the tetrahedra in the polyhedron.
	"""
	cdef int i, k
	cdef double centx = 0.0
	cdef double centy = 0.0
	cdef double centz = 0.0
	for i in range(nvertices):
		centx += vertices[3*i]
		centy += vertices[3*i+1]
		centz += vertices[3*i+2]
	centx /= nvertices
	centy /= nvertices
	centz /= nvertices
	cdef double[:] new_vertices = np.zeros(3*nvertices, dtype=np.float64)
	for i in range(nvertices):
		new_vertices[3*i] = vertices[3*i] - centx
		new_vertices[3*i+1] = vertices[3*i+1] - centy
		new_vertices[3*i+2] = vertices[3*i+2] - centz
	
	cdef double volume = 0.0
	cdef int fid1, fid2, fid3
	cdef double v1x, v1y, v1z
	cdef double v2x, v2y, v2z
	cdef double v3x, v3y, v3z
	cdef double v4x, v4y, v4z
	cdef double nx, ny, nz
	for i in range(num_faces):
		# Algorithm follows that of Dobrovolskis (1996)
		fid1 = face_ids[3*i]
		fid2 = face_ids[3*i+1]
		fid3 = face_ids[3*i+2]
		v1x = new_vertices[3*fid1]
		v1y = new_vertices[3*fid1+1]
		v1z = new_vertices[3*fid1+2]
		v2x = new_vertices[3*fid2]
		v2y = new_vertices[3*fid2+1]
		v2z = new_vertices[3*fid2+2]
		v3x = new_vertices[3*fid3]
		v3y = new_vertices[3*fid3+1]
		v3z = new_vertices[3*fid3+2]
		for k in range(nvertices):
			if k != fid1 and k != fid2 and k != fid3:
				v4x = new_vertices[3*k]
				v4y = new_vertices[3*k+1]
				v4z = new_vertices[3*k+2]
				nx = (v2y-v1y)*(v3z-v1z) - (v2z-v1z)*(v3y-v1y)
				ny = (v2z-v1z)*(v3x-v1x) - (v2x-v1x)*(v3z-v1z)
				nz = (v2x-v1x)*(v3y-v1y) - (v2y-v1y)*(v3x-v1x)
				d = nx*(v4x-v1x) + ny*(v4y-v1y) + nz*(v4z*v1z)
				if d <= 0:
					nx *= -1.0
					ny *= -1.0
					nz *= -1.0
				break
		volume += np.abs(v1x*nx + v1y*ny + v1z*nz)/6	
	return new_vertices, volume

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef getforce(double[:] pos, double[:] angles, double[:] massdens, double[:] semiaxes, int[:] nfaces, int[:] nverts,
 double[:] vertices, int[:] face_ids, int FM_check, double G_grav, double eabs=1e-8, double erel=1e-8, int qval=6, int roundoff=0):
	"""
	Computes the forces/torques on one of the bodies (other body has -F by Newtons 3rd Law)
	Used as a port between C and python call.
	The surface of body 1 is being integrated over, in the potential field of body 2.
	Applicable for bodies of arbitrary shapes

	Input:
	-- pos      (double/float): Position of the two bodies given in order [x1, y1, z1, x2, y2, z2]

	-- angles   (double/float): Rotation angles of the bodies [phi1, theta1, psi1, phi2, theta2, psi2]
							    Rotation angles follow the z-y-x convention (Tait-Bryan)
							    Must not be Euler parameters/Quarternions

	-- massdens (double/float): Mass and densities of the bodies [m1, m2, rho1, rho2]

	-- semiaxes (double/float): Ellipsoid semiaxes given as [a1, b1, c1, a2, b2, c2]
								Semiaxes are ignored if the body is a polyhedron
								Semiaxes must have shapes
								a > b > c
								or 
								a = b > c
								or 
								a = b < c
								else treated as a point mass

	-- nfaces   (int): The number of faces of the polyhedron [n1, n2]. Treated as ellipsoid if n = 0
	-- nverts   (int): Number of vertices of each body [N1, N2]. 

	-- vertices (double/float): The vertices of the polyhedron, ordered as:
								[vx1_1, vy1_1, vz1_1, vx1_2, vy1_2, vz1_2, ..., vx1_N1, vy1_N1, vz1_N1,
								 vx2_1, vx2_1, vz2_1, vx2_2, vx2_2, vz2_2, ..., vx2_N2, vy2_N2, vx2_N2]
								That is, vertices of body 1 in order [x,y,z,x,y,z,...], then following for body 2.
								Number of vertices must be equal 3*nverts
								See documentation for details
	
	-- face_ids (int): Vertex indices that make up each face of the polyhedron on each body.
					   Structure similar to to vertices, for instance
					   [ v1, v2, v3,  v1, v3, v4, ..., w1, w2, w3,   w1, w3, w4, ...]
					   |--face A1--| |--face A2--|    |--face B1--| |--face B2--|
					   See documentation for details

	-- FM_check (int): If set to 1, computes the forces. 
					   If set to 2, computes the torques.

	-- G_grav   (double/float): The gravitational constant. Units must be consistent with input positions.

	Optional:
	-- eabs (double/float): Absolute error limit for integration
	-- erel (double/float): Relative error limit for integration
	-- qval (int): Integration key
	-- roundoff (int): If set to 1, force values smaller than 10^(-16) are rounded down to zero.
	"""

	if qval > 6 or qval < 0:
		raise ValueError("qval must be greater than 0 or less than 6.")
	if FM_check != 1 and FM_check != 2:
		raise ValueError("FM_check must be 1 (force) or 2 (torque).")

	cdef double xc = pos[3] - pos[0]
	cdef double yc = pos[4] - pos[1]
	cdef double zc = pos[5] - pos[2]
	cdef double[11] inpos = [xc, yc, zc, angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], 0.0, 0.0]
	cdef double[6] saxes = [semiaxes[0], semiaxes[1], semiaxes[2], semiaxes[3], semiaxes[4], semiaxes[5]]
	cdef int total_vertices = nverts[0] + nverts[1]
	cdef int[7] vertexids = [nfaces[0], nfaces[1], 0, 1, nverts[0], nverts[1], total_vertices]
	
	cdef double** vertices_in = <double **>malloc(total_vertices*sizeof(double))
	cdef int i, j
	for i in range(total_vertices):
		vertices_in[i] = <double *>malloc(3*sizeof(double))
		for j in range(3):
			vertices_in[i][j] = vertices[3*i+j]

	cdef int* face_ids_other = <int *>malloc(3*nfaces[1]*sizeof(int))
	for i in range(nfaces[1]):
		face_ids_other[3*i] = face_ids[3*i + 3*nfaces[0]]
		face_ids_other[3*i+1] = face_ids[3*i+1 + 3*nfaces[0]]
		face_ids_other[3*i+2] = face_ids[3*i+2 + 3*nfaces[0]]
	cdef int[3] vertex_surface = [0, 0, 0]
	cdef double[3] itol = [eabs, erel, qval]  # Certain danger in converting qval from int to double here...
	# Issues with surface integration with large numbers
	# Masses are scaled so magnitude order is roughly at unity
	cdef double[:] masses = np.array([massdens[0], massdens[1]], dtype=np.float64)
	"""
	### This part of the code is an old implementation which computes the mass 
	### of some arbitrary body based on the density.
	### This is commented out as it can be time consuming to run this multiple times
	### for polyhedral shapes.
	cdef double[:] v_in_temp
	cdef int[:] face_in
	for i in range(2):
		if nverts[i] > 0:
			v_in_temp = np.zeros(3*nverts[i], dtype=np.float64)
			face_in = np.zeros(3*nfaces[i], dtype=np.int32)
			for j in range(3*nverts[i]):
				v_in_temp[j] = vertices[j + 3*nverts[0]*i]
			for j in range(3*nfaces[i]):
				face_in[j] = face_ids[j + 3*nfaces[0]*i]
			v_in_temp, volume = polyhedron_volume(v_in_temp, nverts[i], nfaces[i], face_in)
			for j in range(nverts[i]):
				vertices_in[j + nverts[0]*i][0] = v_in_temp[3*j]
				vertices_in[j + nverts[0]*i][1] = v_in_temp[3*j+1]
				vertices_in[j + nverts[0]*i][2] = v_in_temp[3*j+2]
			masses[i] = rho[i]*volume
		else:
			masses[i] = rho[i]*4*np.pi*semiaxes[3*i]*semiaxes[3*i+1]*semiaxes[3*i+2]/3
	"""
	cdef double mass_scale = 10**np.max(np.floor(np.log10(np.abs(masses))))
	cdef double prefac = massdens[2]*massdens[3]*G_grav/mass_scale

	cdef double Fx = 0.0
	cdef double Fy = 0.0
	cdef double Fz = 0.0
	# The two arrays below are only used for point masses
	cdef double[3] F_pm = [0.0, 0.0, 0.0]
	cdef double[4] m_array = [massdens[0], massdens[1], massdens[3], G_grav]
	
	if nfaces[0] > 0:
		# Is a polyhedron
		for i in range(nfaces[0]):
			vertex_surface[0] = face_ids[3*i]
			vertex_surface[1] = face_ids[3*i+1]
			vertex_surface[2] = face_ids[3*i+2]
			Fx += Force_polyhedron(inpos, itol, saxes, vertices_in, FM_check, 1, 0, vertex_surface, prefac, vertexids, face_ids_other)
			Fy += Force_polyhedron(inpos, itol, saxes, vertices_in, FM_check, 2, 0, vertex_surface, prefac, vertexids, face_ids_other)
			Fz += Force_polyhedron(inpos, itol, saxes, vertices_in, FM_check, 3, 0, vertex_surface, prefac, vertexids, face_ids_other)
	else:
		if np.abs(semiaxes[0]-semiaxes[1]) < 1e-15 and  np.abs(semiaxes[1]-semiaxes[2]) < 1e-15:
			# Is a sphere/point mass
			# Torque on spheres are automatically zero
			if FM_check == 1:
				Force_point_mass(inpos, saxes, 0, vertices_in, vertexids, face_ids_other, F_pm, m_array);
				Fx = F_pm[0]
				Fy = F_pm[1]
				Fz = F_pm[2]
		else:
			# Is an ellipsoid
			Fx = Force_ellipsoid(inpos, itol, saxes, vertices_in, FM_check, 1, 0, prefac, face_ids_other, vertexids);
			Fy = Force_ellipsoid(inpos, itol, saxes, vertices_in, FM_check, 2, 0, prefac, face_ids_other, vertexids);
			Fz = Force_ellipsoid(inpos, itol, saxes, vertices_in, FM_check, 3, 0, prefac, face_ids_other, vertexids);
	
	for i in range(total_vertices):
		free(vertices_in[i])
	free(vertices_in)
	free(face_ids_other)

	if roundoff >= 1:
		if np.abs(Fx) < 1e-16:
			Fx = 0
		if np.abs(Fy) < 1e-16:
			Fy = 0
		if np.abs(Fz) < 1e-16:
			Fz = 0
	# Rescale back to proper units
	Fx *= mass_scale
	Fy *= mass_scale
	Fz *= mass_scale

	return Fx, Fy, Fz

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef getpotential(double[:] pos, double[:] angles, double[:] massdens, double[:] semiaxes, int[:] nfaces, int[:] nverts,
	 double[:] vertices, int[:] face_ids, double G_grav, double eabs=1e-8, double erel=1e-8, int qval=6):
	"""
	Computes the mutual potential between two bodies.
	Used as a port between C and python call.
	The surface of body 1 is being integrated over, in the potential field of body 2.
	Applicable for bodies of arbitrary shape.

	Input:
	-- pos      (double/float): Position of the two bodies given in order [x1, y1, z1, x2, y2, z2]

	-- angles   (double/float): Rotation angles of the bodies [phi1, theta1, psi1, phi2, theta2, psi2]
							    Rotation angles follow the z-y-x convention (Tait-Bryan)
							    Must not be Euler parameters/Quarternions

	-- massdens (double/float): Mass and densities of the bodies [m1, m2, rho1, rho2]

	-- semiaxes (double/float): Ellipsoid semiaxes given as [a1, b1, c1, a2, b2, c2]
								Semiaxes are ignored if the body is a polyhedron
								Semiaxes must have shapes
								a > b > c
								or 
								a = b > c
								or 
								a = b < c
								else treated as a point mass

	-- nfaces   (int): The number of faces of the polyhedron [n1, n2]. Treated as ellipsoid if n = 0
	-- nverts   (int): Number of vertices of each body [N1, N2]. 

	-- vertices (double/float): The vertices of the polyhedron, ordered as:
								[vx1_1, vy1_1, vz1_1, vx1_2, vy1_2, vz1_2, ..., vx1_N1, vy1_N1, vz1_N1,
								 vx2_1, vx2_1, vz2_1, vx2_2, vx2_2, vz2_2, ..., vx2_N2, vy2_N2, vx2_N2]
								That is, vertices of body 1 in order [x,y,z,x,y,z,...], then following for body 2.
								Number of vertices must be equal 3*nverts
								See documentation for details
	
	-- face_ids (int): Vertex indices that make up each face of the polyhedron on each body.
					   Structure similar to to vertices, for instance
					   [ v1, v2, v3,  v1, v3, v4, ..., w1, w2, w3,   w1, w3, w4, ...]
					   |--face A1--| |--face A2--|    |--face B1--| |--face B2--|
					   See documentation for details

	-- G_grav   (double/float): The gravitational constant. Units must be consistent with input positions.

	Optional:
	-- eabs (double/float): Absolute error limit for integration
	-- erel (double/float): Relative error limit for integration
	-- qval (int): Integration key
	"""

	if qval > 6 or qval < 0:
		raise ValueError("qval must be greater than 0 or less than 6.")

	cdef double xc = pos[3] - pos[0]
	cdef double yc = pos[4] - pos[1]
	cdef double zc = pos[5] - pos[2]
	cdef double[11] inpos = [xc, yc, zc, angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], 0.0, 0.0]
	cdef double[6] saxes = [semiaxes[0], semiaxes[1], semiaxes[2], semiaxes[3], semiaxes[4], semiaxes[5]]
	cdef int total_vertices = nverts[0] + nverts[1]
	cdef int[6] vertexids = [nverts[0], nverts[1], nfaces[0], nfaces[1], 0, 1]
	cdef double* vertices_in = <double *>malloc(3*total_vertices*sizeof(double))
	cdef int i
	for i in range(3*total_vertices):
		vertices_in[i] = vertices[i]

	cdef int* face_ids_other = <int *>malloc(3*nfaces[1]*sizeof(int))
	for i in range(nfaces[1]):
		face_ids_other[3*i] = face_ids[3*i + 3*nfaces[0]]
		face_ids_other[3*i+1] = face_ids[3*i+1 + 3*nfaces[0]]
		face_ids_other[3*i+2] = face_ids[3*i+2 + 3*nfaces[0]]
	cdef int[3] vertex_surface = [0, 0, 0]
	cdef double[3] itol = [eabs, erel, qval]  # Certain danger in converting qval from int to double here...
	
	# Issues with surface integration with large numbers
	# Masses are scaled so magnitude order is roughly at unity
	cdef double[:] masses = np.array([massdens[0], massdens[1]], dtype=np.float64)
	cdef double mass_scale = 10**np.max(np.floor(np.log10(np.abs(masses))))
	cdef double U = 0.0
	cdef double[4] m_array = [massdens[2]/mass_scale, massdens[3]/mass_scale, massdens[1]/mass_scale, G_grav*mass_scale]
	# For ellipsoid
	cdef double[4] limits = [-semiaxes[2], semiaxes[2], 0.0, 2*pi]
	if nfaces[0] > 0:
		# Is a polyhedron
		for i in range(nfaces[0]):
			vertex_surface[0] = face_ids[3*i]
			vertex_surface[1] = face_ids[3*i+1]
			vertex_surface[2] = face_ids[3*i+2]
			U += mutual_potential_polyhedron(saxes, inpos, itol, vertices_in, vertex_surface, m_array, face_ids_other, vertexids, 0)
	else:
		if np.abs(semiaxes[0]-semiaxes[1]) < 1e-15 and  np.abs(semiaxes[1]-semiaxes[2]) < 1e-15:
			# Is a sphere/point mass
			U = mutual_potential_point_mass(saxes, inpos, vertices_in, m_array, face_ids_other, vertexids, 0)
		else:
			# Is an ellipsoid
			U = mutual_potential_ellipsoid(limits, saxes, inpos, itol, vertices_in, vertexids, face_ids_other, m_array, 0)

	free(vertices_in)
	free(face_ids_other)

	# Rescale back to proper units and takes into account the minus sign
	U *= -mass_scale
	return U
