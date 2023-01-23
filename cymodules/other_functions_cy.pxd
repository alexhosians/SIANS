import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport pi
import sys

cdef extern from "SurfaceIntegrals.h":
	double Force_ellipsoid(double input[11], double itol[3], double semiaxes[6], double **vertices, int FM_check,
			int component, int eulerparam, double prefac, int* face_ids_other, int vertexids[7]);

	double Force_polyhedron(double input[11], double itol[3], double semiaxes[6], double **vertices,
			int FM_check, int component, int eulerparam, int vertex_combo[3], double prefac, int vertexids[7], int *face_ids_other);

	int Force_point_mass(double input[11], double semiaxes[6], int eulerparam, double **vertices, int vertexids[7],
	 		int *face_ids_other, double result[3], double mass_array[4]);

	double mutual_potential_ellipsoid(double limits[4], double semiaxes[6], double input_values[11],
 			double itols[3], double *vertices1D, int vertexids[6], int* face_ids, double massdens[4], int eulerparams)

	double mutual_potential_polyhedron(double semiaxes[6], double input_values[11], double itols[3], 
			double *vertices1D, int vertex_combo[3], double massdens[4], int* face_ids, int vertexids[6], int eulerparam)

	double mutual_potential_point_mass(double semiaxes[6], double input_values[11], double *vertices1D,
			double massdens[4], int* face_ids_other, int vertexids[6], int eulerparam)

cdef inline double moment_inertia_ellipsoid(double a,double b,double c,double rho,int i):
	""" 
	Values of the moment of inertia for ellipsoids
	Returns a C-variable instead of python object
	"""
	cdef double M = (4.0/15.0)*pi*rho*a*b*c
	cdef double mom_intert
	if i == 1:
		mom_intert = M*(b*b+c*c)
		return mom_intert
	elif i == 2:
		mom_intert = M*(a*a+c*c)
		return mom_intert
	elif i == 3:
		mom_intert = M*(a*a+b*b)
		return mom_intert
	else:
		raise ValueError("Index i for moment_inertia_spheroid() not set properly.")

cdef inline int moment_inertia_ellipsoid_matrix(double a, double b, double c, double rho, double[:,:] I):
	"""
	Computes moment of inertia for an ellipsoid, full matrix.
	"""
	cdef double M = (4.0/15.0)*pi*rho*a*b*c
	I[0][0] = M*(b*b+c*c)
	I[1][1] = M*(a*a+c*c)
	I[2][2] = M*(a*a+b*b)
	I[0][1] = 0
	I[0][2] = 0
	I[1][0] = 0
	I[1][2] = 0
	I[2][0] = 0
	I[2][1] = 0
	return 0

cdef inline double[:] moment_inertia_polyhedron(double* vertices, int[:] index_combo, int N_faces, int N_vertices, double mass, int id_fac):
	cdef double[:] volume_elements = np.zeros(N_faces, dtype=np.float64)
	cdef double[:] vertices_in = np.zeros(12, dtype=np.float64)
	cdef double[:] n_i
	cdef int i, j, k, l
	cdef int v1, v2, v3
	cdef double volume = 0
	cdef double[:] P = np.zeros(9, dtype=np.float64)
	cdef double[:] D = np.zeros(3, dtype=np.float64)
	cdef double[:] E = np.zeros(3, dtype=np.float64)
	cdef double[:] F = np.zeros(3, dtype=np.float64)
	cdef double prefac, rho
	cdef double[:] I = np.zeros(9, dtype=np.float64)
	with cython.boundscheck(False):
		for i in range(N_faces):
			v1 = index_combo[3*i]
			v2 = index_combo[3*i+1]
			v3 = index_combo[3*i+2]
			for l in range(N_vertices):
				if l != v1 and l != v2 and l != v3:
					for j in range(3):
						vertices_in[j] = vertices[id_fac+3*v1+j]
						vertices_in[j+3] = vertices[id_fac+3*v2+j]
						vertices_in[j+6] = vertices[id_fac+3*v3+j]
						vertices_in[j+9] = vertices[id_fac+3*l+j]
					n_i = normal_vector_face_summed(vertices_in)
					break
			volume_elements[i] = (n_i[0]*vertices[id_fac+3*v1] + n_i[1]*vertices[id_fac+3*v1+1] + n_i[2]*vertices[id_fac+3*v1+2])/6
		for i in range(N_faces):
			volume += volume_elements[i]
		rho = mass/volume
		for i in range(N_faces):
			prefac = rho*volume_elements[i]/20
			v1 = index_combo[3*i]
			v2 = index_combo[3*i+1]
			v3 = index_combo[3*i+2]
			for l in range(3):
				D[l] = vertices[id_fac + 3*v1 + l]
				E[l] = vertices[id_fac + 3*v2 + l]
				F[l] = vertices[id_fac + 3*v3 + l]
			for j in range(3):
				for k in range(3):
					P[3*j + k] += prefac*(2*(D[j]*D[k] + E[j]*E[k] + F[j]*F[k]) + D[j]*E[k] + D[k]*E[j] +\
									   D[j]*F[k] + D[k]*F[j] + E[j]*F[k] + E[k]*F[j])

		I[0] = P[4] + P[8]
		# 01 and 02 swapped here...
		I[1] = -P[2]
		I[2] = -P[1]
		# 10 and 20 swapped here...
		I[3] = -P[6]
		I[4] = P[0] + P[8]
		I[5] = -P[5]
		# and here
		I[6] = -P[3]
		I[7] = -P[7]
		I[8] = P[4] + P[0]
	return I

cdef inline double[:] normal_vector_face_summed(double[:] vertices):
	"""
	Returns normal vector of a triangular face on a polyhedron.
	"""
	cdef double r2r1x = vertices[3] - vertices[0]
	cdef double r2r1y = vertices[4] - vertices[1]
	cdef double r2r1z = vertices[5] - vertices[2]	
	cdef double r3r1x = vertices[6] - vertices[0]
	cdef double r3r1y = vertices[7] - vertices[1]
	cdef double r3r1z = vertices[8] - vertices[2]	
	cdef double r4r1x = vertices[9] - vertices[0]
	cdef double r4r1y = vertices[10] - vertices[1]
	cdef double r4r1z = vertices[11] - vertices[2]	

	cdef double cprodx = r3r1y*r2r1z - r3r1z*r2r1y
	cdef double cprody = r3r1z*r2r1x - r3r1x*r2r1z
	cdef double cprodz = r3r1x*r2r1y - r3r1y*r2r1x

	cdef double d = cprodx*r4r1x + cprody*r4r1y + cprodz*r4r1z
	cdef double[:] n_i = np.zeros(3, dtype=np.float64)
	if d < 0:
		n_i[0] = cprodx
		n_i[1] = cprody
		n_i[2] = cprodz
	else:
		n_i[0] = -cprodx
		n_i[1] = -cprody
		n_i[2] = -cprodz
	return n_i

cdef inline double polyhedron_radius(double[:] vertices, int n_vertices):
	cdef int i
	cdef double radius = 0
	cdef double temp_rad = 0
	with cython.boundscheck(False):
		for i in range(n_vertices):
			temp_rad = np.sqrt(vertices[3*i]**2 + vertices[3*i+1]**2 + vertices[3*i+2]**2)
			if temp_rad > radius:
				radius = temp_rad
	return radius

