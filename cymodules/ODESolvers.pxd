# cython: linetrace=True
# Basic imports
import sys
# Numpy and scipy.integrate import
import numpy as np
cimport numpy as np
cimport cython
#from cython.parallel import prange
# Import C square root call
from libc.math cimport pi, sqrt, fabs
from other_functions_cy cimport moment_inertia_ellipsoid
from other_functions_cy cimport moment_inertia_polyhedron as I_poly_pyx
from other_functions_cy cimport polyhedron_radius

cdef extern from "commonCfuncs.h":
	int ellipsoid_intersect_check(double semiaxes[6], double input[11], double positions[6], int eulerparam)

cdef extern from "diffeqsolve.h":
	int ode_solver_2body(double t, double* y, double* dfdt, double* params_input, double* semiaxes, double itol[3], double *vertices1D,
						int* face_ids, double* moment_inertia, int triggers[3])
	int ode_solver_Nbody(double t, double* y, double* dfdt, double* params_input, double* semiaxes, double itol[3], double *vertices1D,
 						 int *face_ids, double *moment_inertia, int triggers[3])
	
cdef inline int cross_product(double[:] v1, double[:] v2, double[:] result) nogil:
	""" Function computing cross product of two vectors """
	with cython.boundscheck(False):
		result[0] = v1[1]*v2[2] - v1[2]*v2[1]
		result[1] = v1[2]*v2[0] - v1[0]*v2[2]
		result[2] = v1[0]*v2[1] - v1[1]*v2[0]

cdef inline double dot_product(double[:] v1, double[:] v2) nogil:
	cdef double dprod
	with cython.boundscheck(False):
		dprod = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
	return dprod

cdef inline double[:] Step_solution_RK(double[:] y, double[:,:] K, double[:] B, int n_stages, int numvars, double dt):
	""" 
	Function that steps the solution based on the K-parameters from the Runge-Kutta solver.
	"""
	cdef double[:] y_new = np.zeros(numvars, dtype=np.float64)
	cdef double new_steps = 0.0
	cdef int j, k
	with cython.boundscheck(False):
		for k in range(numvars):
			new_steps = 0.0
			for j in range(n_stages):
				new_steps += K[j,k]*B[j]
			y_new[k] = y[k] + new_steps*dt
	return y_new

cdef inline double Error_Hairer(double[:] y, double[:] y_new, double[:] y_tild, int numvars, double atol, double rtol):
	"""
	Function that computes the local error.
	Based on the book of Harier, Norsett and Wanner.
	https://www.springer.com/gp/book/9783540566700
	"""
	cdef double sc, tau
	cdef double error_temp = 0.0
	cdef int i
	# Compute the error tolerance
	with cython.boundscheck(False):
		for i in range(numvars):
			sc = atol + max(fabs(y[i]), fabs(y_new[i]))*rtol
			tau = y_new[i] - y_tild[i]
			error_temp += (tau/sc)**2
	cdef double error = sqrt(error_temp/numvars)
	return error

cdef inline double Error_Tsitouras(double[:] y_new, double[:] y_tild, int numvars):
	"""
	Function that computes the local error.
	Based on the paper from Ch. Tsitouras (2011):
	https://doi.org/10.1016/j.camwa.2011.06.002.
	"""
	#cdef double[:] tau = np.zeros(numvars, dtype=np.float64)
	cdef double tau
	cdef double error_temp = 0.0
	cdef int i
	with cython.boundscheck(False):
		for i in range(numvars):
			#tau[i] = y_new[i] - y_tild[i]
		# Norm squared of the vector above
		#for i in range(numvars):
			tau = y_new[i] - y_tild[i]
			error_temp += tau*tau
	cdef double E_n = sqrt(error_temp)
	return E_n

cdef inline double errfac_Feagin(int n_stages):
	""" 
	Returns different error factor based on Feagin RK method 
	For order 10(8), returns 1/360
	For order 12(10), returns 49/640
	"""
	cdef double factor
	if n_stages == 17:
		factor = 1.0/360
	else:
		factor = 49.0/640
	return factor

cdef inline double Error_adaptive(double[:] y, double[:] y_new, double[:] y_tild, int numvars, double atol, double rtol,
 int errortype):
	cdef double sc, tau
	cdef double error_temp = 0.0
	cdef int i
	cdef double error
	if errortype == 1:
		# Hairer
		error = Error_Hairer(y, y_new, y_tild, numvars, atol, rtol)
	elif errortype == 2:
		# Tsitouras
		with cython.boundscheck(False):
			for i in range(numvars):
				tau = y_new[i] - y_tild[i]
				error_temp += tau*tau
		error = sqrt(error_temp)
	return error

cdef inline int check_reduced_step(double localErrorNorm, double tolerance, int errorchecks):
	if errorchecks == 1:
		if localErrorNorm > 1.0:
			return 1
		else:
			return 0
	elif errorchecks == 2:
		if localErrorNorm >= tolerance:
			return 1
		else:
			return 0
	elif errorchecks == 3:
		if localErrorNorm > tolerance:
			return 1
		else:
			return 0
				
cdef inline double rescale_stepsize(double h_old, double localErrorNorm, double fac, double p_order_fac, double tolerance, 
	double hmin, double hmax, int errorchecks):
	cdef double h
	if errorchecks == 1:
		# Hairer error
		h = h_old*min(hmax, max(hmin, fac*(1.0/localErrorNorm)**(p_order_fac)))
	elif errorchecks == 2:
		# Tsitouras error
		h = fac*h_old*(tolerance/localErrorNorm)**(p_order_fac)
	else:
		# Faegin error
		h = h_old*min(hmax, max(fac*(tolerance/(localErrorNorm))**(p_order_fac), hmin))
	return h

cdef inline int normal_vector_face_summed(int v1, int v2, int v3, int v4, double[:] vertices, double[:] n_i):
	cdef double[:] r2r1 = np.zeros(3, dtype=np.float64)
	cdef double[:] r3r1 = np.zeros(3, dtype=np.float64)
	cdef double[:] r4r1 = np.zeros(3, dtype=np.float64)
	cdef int i
	for i in range(3):
		r2r1[i] = vertices[3*v2 + i] - vertices[3*v1 + i]
		r3r1[i] = vertices[3*v3 + i] - vertices[3*v1 + i]
		r4r1[i] = vertices[3*v4 + i] - vertices[3*v1 + i]
	
	cdef double[:] cross_prod = np.zeros(3, dtype=np.float64)
	cross_product(r3r1, r2r1, cross_prod)
	cdef double	d = dot_product(cross_prod, r4r1)
	if d < 0:
		n_i[0] = cross_prod[0]
		n_i[1] = cross_prod[1]
		n_i[2] = cross_prod[2]	
	else:
		n_i[0] = -cross_prod[0]
		n_i[1] = -cross_prod[1]
		n_i[2] = -cross_prod[2]
	return 0

cdef inline double[:] normal_vector_triangle_righthand(int[:] indices, double[:] vertices):
	cdef int i
	cdef int setup = 0
	cdef int flip = 0
	cdef int v1 = indices[0]
	cdef int v2 = indices[1]
	cdef int v3 = indices[2]
	cdef int v4 = indices[3]
	cdef double[:] r2r1 = np.zeros(3, dtype=np.float64)
	cdef double[:] r3r1 = np.zeros(3, dtype=np.float64)
	cdef double[:] r4r1 = np.zeros(3, dtype=np.float64)
	cdef double[:] cross_prod = np.zeros(3, dtype=np.float64)
	cdef double d
	while setup == 0:
		for i in range(3):
			r2r1[i] = vertices[3*v2 + i] - vertices[3*v1 + i]
			r3r1[i] = vertices[3*v3 + i] - vertices[3*v1 + i]
			r4r1[i] = vertices[3*v4 + i] - vertices[3*v1 + i]
		cross_product(r3r1, r2r1, cross_prod)
		d = dot_product(cross_prod, r4r1)
		if d <= 0:
			setup = 1
		else:
			flip = 1
			v1 = indices[2]
			v3 = indices[0]
	if flip == 1:
		indices[0] = v1
		indices[2] = v3
	return cross_prod

cdef inline int normal_dot_check(int M, double[:,:] N_faces):
	cdef double dot_prod_normal_vecs
	for i in range(M):
		for j in range(M-1):
			if (i>j):
				dot_prod_normal_vecs = N_faces[i][0]*N_faces[j][0] + N_faces[i][1]*N_faces[j][1] + N_faces[i][2]*N_faces[j][2]
				if dot_prod_normal_vecs <= 0:
					return 0
	return 1

cdef inline int[:] Face_id_combinations_pyx(double[:] vertices, int N_vertices, int[:] N_faces):
	cdef int i, j, k, l, m, n
	cdef int M = N_vertices - 3
	cdef int N_extra_normals = (M-1)*int(M/2)
	cdef int N_faces_local = 0
	cdef int count
	cdef int[:] face_ids, indices
	cdef double[:,:] N_vectors, r_vectors
	cdef double[:] normal_vector
	cdef int allocate, is_segment, counter, unique_ids
	cdef int[:] face_ids_current = np.zeros(3, dtype=np.int32)
	if N_extra_normals == 0:
		N_faces[0] = 4
		count = 0
		#face_ids = <int *>malloc(12*sizeof(int))
		face_ids = np.zeros(12, dtype=np.int32)
		for i in range(2):
			for j in range(1,3):
				for k in range(2,4):
					if i<j and j<k:
						face_ids[3*count] = i
						face_ids[3*count+1] = j
						face_ids[3*count+2] = k
						count += 1
	else:
		N_vectors = np.zeros((M,3), dtype=np.float64)
		face_ids = np.zeros(3, dtype=np.int32)
		allocate = 0
		for i in range(N_vertices):
			for j in range(N_vertices):
				for k in range(N_vertices):
					if i < j and j < k:
						counter = 0
						for l in range(N_vertices):
							indices = np.array([i, j, k, l], dtype=np.int32)
							if len(np.unique(indices)) == len(indices):
								N_vectors[counter] = normal_vector_triangle_righthand(indices, vertices)
								counter += 1

						is_segment = normal_dot_check(M, N_vectors)
						if is_segment == 1:
							N_faces_local += 1
							if allocate:
								face_ids_current[0] = indices[0]
								face_ids_current[1] = indices[1]
								face_ids_current[2] = indices[2]
								face_ids = np.concatenate([face_ids, face_ids_current])
							else:
								face_ids[0] = indices[0]
								face_ids[1] = indices[1]
								face_ids[2] = indices[2]
							allocate = 1
		N_faces[0] = N_faces_local
	return face_ids

cdef inline int Sort_mainprop_pointer_polyhedron(double* main_prop_ptr, double* vertices, int NumBodies, 
	int[:] p_info, int[:] face_ids, double[:] masses, double[:] semiaxes, double* moment_inertia):
	"""
	Computes volumes, centroids, moment of inertia and densities of the bodies.
	Centers the vertices around the centroid.
	"""
	cdef double[:] vertices_in
	cdef double volume
	cdef int num_faces = 0
	cdef int total_faces = 0
	cdef int id_fac = 0
	cdef int faces_iterated = 0
	cdef double[:] centroid
	cdef double[:] cprod = np.zeros(3, dtype=np.float64)
	cdef double[:] ad = np.zeros(3, dtype=np.float64)
	cdef double[:] bd = np.zeros(3, dtype=np.float64)
	cdef double[:] cd = np.zeros(3, dtype=np.float64)
	cdef double[:] n_i = np.zeros(3, dtype=np.float64)
	cdef int[:] num_faces_arr = np.zeros(1, dtype=np.int32)
	cdef int[:] face_ids_in
	cdef double[:] I_tensor
	cdef int i, j, k
	cdef int v1, v2, v3
	cdef int v1f, v2f, v3f
	cdef int i3, i9
	cdef int is_sphere
	cdef double radius
	for i in range(NumBodies):
		radius = 0
		is_sphere = 0
		centroid = np.zeros(3, dtype=np.float64)
		nvertices = p_info[i]
		num_faces = p_info[NumBodies+i]
		face_ids_in = np.zeros(3*num_faces, dtype=np.int32)
		for j in range(3*num_faces):
			face_ids_in[j] = face_ids[j + faces_iterated]
		faces_iterated += 3*num_faces
		i3 = 3*i
		i9 = 9*i
		if nvertices > 0:
			# Center vertices around centroid
			vertices_in = np.zeros(nvertices*3, dtype=np.float64)
			for j in range(3*nvertices):
				vertices_in[j] = vertices[id_fac + j]
			radius = polyhedron_radius(vertices_in, nvertices)
			if num_faces == 4:
				# Volume for tetrahedron. Known
				for j in range(3):
					ad[j] = vertices_in[j] - vertices_in[9+j]
					bd[j] = vertices_in[3+j] - vertices_in[9+j]
					cd[j] = vertices_in[6+j] - vertices_in[9+j]
				cross_product(bd, cd, cprod)
				volume = np.abs(dot_product(ad, cprod))/6
				for j in range(4):
					centroid[0] += vertices_in[3*j]/4
					centroid[1] += vertices_in[3*j+1]/4
					centroid[2] += vertices_in[3*j+2]/4
			else:
				# For other general polygon shapes
				volume = 0
				for j in range(num_faces):
					v1 = face_ids_in[3*j]
					v2 = face_ids_in[3*j+1]
					v3 = face_ids_in[3*j+2]
					for k in range(nvertices):
						if k != v1 and k != v2 and k != v3:
							normal_vector_face_summed(v1, v2, v3, k, vertices_in, n_i)
							break
					v1f = 3*v1
					v2f = 3*v2
					v3f = 3*v3
					centroid[0] += n_i[0]*((vertices_in[v1f] + vertices_in[v2f])**2 + (vertices_in[v2f] + vertices_in[v3f])**2 
								 + (vertices_in[v1f] + vertices_in[v3f])**2)/24
					centroid[1] += n_i[1]*((vertices_in[v1f + 1] + vertices_in[v2f + 1])**2
								 + (vertices_in[v2f + 1] + vertices_in[v3f + 1])**2 
								 + (vertices_in[v1f + 1] + vertices_in[v3f + 1])**2)/24
					centroid[2] += n_i[2]*((vertices_in[v1f + 2] + vertices_in[v2f + 2])**2
								 + (vertices_in[v2f + 2] + vertices_in[v3f + 2])**2 
								 + (vertices_in[v1f + 2] + vertices_in[v3f + 2])**2)/24
					volume += (vertices_in[v1f]*n_i[0] + vertices_in[v1f + 1]*n_i[1] + vertices_in[v1f + 2]*n_i[2])/6.0
				centroid[0] /= 2*volume
				centroid[1] /= 2*volume
				centroid[2] /= 2*volume
			for j in range(nvertices):
				vertices[id_fac + 3*j] -= centroid[0]
				vertices[id_fac + 3*j+1] -= centroid[1]
				vertices[id_fac + 3*j+2] -= centroid[2]
			I_tensor = I_poly_pyx(vertices, face_ids_in, num_faces, nvertices, masses[i], id_fac)
			for k in range(9):
				moment_inertia[i9+k] = I_tensor[k]
			id_fac += 3*nvertices
			total_faces += num_faces
				
		else:
			# For ellipsoid
			num_faces = 0
			volume = (4.0/3.0)*pi*semiaxes[i3]*semiaxes[i3+1]*semiaxes[i3+2]
			if volume == 0:
				is_sphere = 1
				moment_inertia[i9] = 0
				moment_inertia[i9+4] = 0
				moment_inertia[i9+8] = 0
			else:
				moment_inertia[i9] = moment_inertia_ellipsoid(semiaxes[i3], semiaxes[i3+1], semiaxes[i3+2], masses[i]/volume, 1)
				moment_inertia[i9+4] = moment_inertia_ellipsoid(semiaxes[i3], semiaxes[i3+1], semiaxes[i3+2], masses[i]/volume, 2)
				moment_inertia[i9+8] = moment_inertia_ellipsoid(semiaxes[i3], semiaxes[i3+1], semiaxes[i3+2], masses[i]/volume, 3)
			moment_inertia[i9+1] = 0
			moment_inertia[i9+2] = 0
			moment_inertia[i9+3] = 0
			moment_inertia[i9+5] = 0
			moment_inertia[i9+6] = 0
			moment_inertia[i9+7] = 0
			if semiaxes[i3] >= semiaxes[i3+1] and semiaxes[i3] >= semiaxes[i3+2]:
				radius = semiaxes[i3]
			elif semiaxes[i3+1] >= semiaxes[i3] and semiaxes[i3+1] >= semiaxes[i3+2]:
				radius = semiaxes[i3+1]
			elif semiaxes[i3+2] >= semiaxes[i3] and semiaxes[i3+2] >= semiaxes[i3+1]:
				radius = semiaxes[i3+2]
			else:
				radius = 0

		if volume <= 0 and is_sphere == 0:
			print("Volume is either zero or negative!")
			sys.exit(0)
		main_prop_ptr[i] = masses[i]
		if is_sphere:
			main_prop_ptr[i+NumBodies] = 0
		else:
			main_prop_ptr[i+NumBodies] = masses[i]/volume
		main_prop_ptr[i+NumBodies*2] = radius
		main_prop_ptr[i+NumBodies*3] = num_faces
	#free(face_ids_in_ptr)
	return 0

cdef inline int ellipsoid_intersect_check_pyx(double[:] y, double[:] semiaxes, int eulerparam):
	cdef double x_A = y[6]
	cdef double y_A = y[7]
	cdef double z_A = y[8]
	cdef double x_B = y[9]
	cdef double y_B = y[10]
	cdef double z_B = y[11]

	cdef double a_A = semiaxes[0]
	cdef double b_A = semiaxes[1]
	cdef double c_A = semiaxes[2]
	cdef double a_B = semiaxes[3]
	cdef double b_B = semiaxes[4]
	cdef double c_B = semiaxes[5]
	cdef double[6] semiaxes_in = [a_A, b_A, c_A, a_B, b_B, c_B]
	cdef double[6] positions = [x_A, y_A, z_A, x_B, y_B, z_B]
	cdef double input_values[11]
	cdef double e0_A, e1_A, e2_A, e3_A, e0_B, e1_B, e2_B, e3_B
	cdef double phi_A, theta_A, psi_A, phi_B, theta_B, psi_B
	if eulerparam:
		e0_A = y[18]
		e1_A = y[19]
		e2_A = y[20]
		e3_A = y[21]
		e0_B = y[22]
		e1_B = y[23]
		e2_B = y[24]
		e3_B = y[25]
		input_values[:] = [0.0, 0.0, 0.0, e0_A, e1_A, e2_A, e3_A, e0_B, e1_B, e2_B, e3_B]
	else:
		phi_A = y[18];
		theta_A = y[19];
		psi_A = y[20];
		phi_B = y[21];
		theta_B = y[22];
		psi_B = y[23];
		input_values[:] = [0.0, 0.0, 0.0, phi_A, theta_A, psi_A, phi_B, theta_B, psi_B, 0.0, 0.0]

	cdef int collide =  ellipsoid_intersect_check(semiaxes_in, input_values, positions, eulerparam)
	return collide