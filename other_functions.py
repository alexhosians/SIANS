#####
# A script containing an assortment of different functions.
# Mostly used to compute smaller stuff, such as potentials and kappa.
# Many of these functions are unused for the main software...
#####

import numpy as np
from scipy.linalg import eig
from scipy.special import ellipeinc, ellipkinc
from scipy import integrate
from scipy.optimize import minimize, fsolve
import sys
# Defining some global pre-factors used in many areas
vol_pre = (4.0/3.0)*np.pi

def kappa_value(a_sq,c_sq,x_sq,y_sq,z_sq,positive_root=True):
	""" 
	Computes the value of kappa as given by Macmillan.
	By default, returns with the square root added and not subtracted.
	The value of kappa must be positive.
	Input values of coordinates and radii must be squared!
	"""
	# Computing expressions for B and C, where A = 1
	B = a_sq + c_sq - x_sq - y_sq - z_sq
	C = a_sq*(c_sq - z_sq) - c_sq*(x_sq + y_sq)
	if positive_root:
		return 0.5*(-B + np.sqrt(B*B - 4.0*C))
	else:
		return 0.5*(-B - np.sqrt(B*B - 4.0*C))


def kappa_derivative(a_sq,c_sq,x,y,z,x_sq,y_sq,z_sq,component,positive_root=True):
	""" 
	Computes the derivative of kappa as given by Macmillan.
	See notes for the derivative expressions.
	By default, returns with the square root added and not subtracted
	Component 1,2,3 corresponds to x,y,z coordinates respectively.
	"""
	# Computing expressions for B and C, where A = 1
	B = a_sq + c_sq - x_sq - y_sq - z_sq
	C = a_sq*(z_sq + c_sq) - c_sq*(x_sq + y_sq)
	if positive_root:
		if component == 1:
			return x*(1.0 + (2.0*c_sq - B)*np.sqrt(B - 4.0*C))
		elif component == 2:
			return y*(1.0 + (2.0*c_sq - B)*np.sqrt(B - 4.0*C))
		elif component == 3:
			return z*(1.0 - (2.0*a_sq + B)*np.sqrt(B - 4.0*C))
		else:
			raise ValueError("Component not set right. Currently component = %g. Try component = 1,2 or 3" %(component))
	else:
		if component == 1:
			return x*(1.0 - (2.0*c_sq - B)*np.sqrt(B - 4.0*C))
		elif component == 2:
			return y*(1.0 - (2.0*c_sq - B)*np.sqrt(B - 4.0*C))
		elif component == 3:
			return z*(1.0 + (2.0*a_sq + B)*np.sqrt(B - 4.0*C))
		else:
			raise ValueError("Component not set right. Currently component = %g. Try component = 1,2 or 3" %(component))

def g_field(a_sq,c_sq,x,y,z,kappa,pre_rho,mass):
	"""
	Computes the gravitational field g = nabla Phi of the gravitational potential.
	See notes for the expressions.
	Component 1,2,3 corresponds to x,y,z coordinates respectively.
	Kappa is assumed to be constant here.
	"""

	a_min_c = a_sq - c_sq
	if a_sq > c_sq: # Oblate
		f1 = np.sqrt(c_sq + kappa)/(a_min_c*(a_sq+kappa))
		f2 = np.arcsin(np.sqrt(a_min_c/(a_sq+kappa)))/a_min_c**(3./2.)
		f3 = 1.0 / (a_min_c*np.sqrt(c_sq + kappa))
		gx = pre_rho*x*(f1-f2)
		gy = pre_rho*y*(f1-f2)
		gz = 2.0*pre_rho*z*(f2-f3)
	elif a_sq < c_sq: # Prolate
		f1 = np.sqrt(c_sq + kappa)/(-a_min_c*(a_sq+kappa))
		f2 = np.arcsin(np.sqrt(-a_min_c/(a_sq+kappa)))/(-a_min_c)**(3./2.)
		f3 = 1.0 / (-a_min_c*np.sqrt(c_sq + kappa))
		gx = pre_rho*x*(f2-f1)
		gy = pre_rho*y*(f2-f1)
		gz = 2.0*pre_rho*z*(f3-f2)
	elif np.abs(a_sq - c_sq) < 1e-13: # Sphere
		denominator = -(x**2 + y**2 + z**2)**(3.0/2.0)	# 1/A
		f1 = x/denominator 
		f2 = y/denominator
		f3 = z/denominator
		gx = f1*mass
		gy = f2*mass
		gz = f3*mass

	return gx,gy,gz

#def Potential_spheroid(x,y,z,a,c,ma,mb,ka):
def Potential_spheroid(x_sq,y_sq,z_sq,a,c,mb,ka):
	""" 
	Computes and returns the potential Phi(x,y,z) of a spheroid (MacMillan 1930)
	Coordinates x,y,z must be the coordinates which forms a vector that points
	from the centroid of the source to the surface of the other object considered.
	E.g. x = xa_surface - xb_center 
	"""
	a_sq = a**2
	c_sq = c**2
	a_sq_c = c*a_sq
	a_min_c = a_sq - c_sq
	#volA = vol_pre*(a_sq)*c
	#rho_A = ma/volA

	xya_sq = x_sq + y_sq
	za_sq = z_sq
	c2ka = c_sq + ka

	# Compute the potential of body A when A is an oblate spheroid (a=b > c)
	if (a > c):
		v1a = 2.0 * np.pi * a_sq_c/np.sqrt(a_min_c)
		v1b = 1.0 - (xya_sq - 2.0 * za_sq)/(2.0*a_min_c)
		v1c = np.arcsin(np.sqrt(a_min_c/(a_sq + ka)))
		v1 = v1a * v1b * v1c	 
		
		v2a = np.pi * a_sq_c * np.sqrt(c2ka)/a_min_c
		v2b = xya_sq/(a_sq + ka)			
		v2 = v2a * v2b
	
		v3a = -np.pi * a_sq_c/a_min_c
		v3b = 2.0 * za_sq/np.sqrt(c2ka)
		v3 = v3a * v3b

	# Compute potential of body A when A is a prolate spheroid (a=b < c)
	elif (a < c):
		ca2 = -a_min_c
		v1a = (2.0 * np.pi * a_sq_c)/np.sqrt(ca2)
		v1b = 1.0 + (xya_sq - 2.0*za_sq)/(2.0*ca2)
		v1c = np.arcsinh( np.sqrt(ca2/(a_sq + ka)) )
		v1 = v1a * v1b * v1c
				
		v2a = np.pi * a_sq_c * np.sqrt(c2ka)/ca2
		v2b = xya_sq/(a_sq + ka)
		v2 = -v2a * v2b
			
		v3a = np.pi * a_sq_c/(ca2)
		v3b = 2.0 * za_sq / np.sqrt(c2ka)
		v3 = v3a * v3b 

	# Compute potential of body A when A is a sphere (a = b = c)
	# Uses the inverse square law, assumes G = 1.0
	elif (a == c):
		v1 = 0
		v2 = 0
		v3 = mb/(np.sqrt(x_sq + y_sq +z_sq))	# Use this when simpsons
		#v3 = mb/(np.linalg.norm(np.array([x, y, z]))) # Use this when dblquad
		#v3 = mb/(np.sqrt(x1_sq + y1_sq + z1_sq)*rho_A)
	
	# potential of B				
	Phi = (v1 + v2 + v3)
	return Phi

def moment_intertia(r,phi,theta,func):
	""" Volume integrand used to compute the moment of inertia """
	Jacobian = r**2*np.sin(phi)
	return func*Jacobian

def get_Rx(phi,transpose=False):
	""" Rotation matrix along x-axis """
	if transpose:
		R_x = np.array([[1,0,0],
						[0, np.cos(phi),np.sin(phi)],
						[0,-np.sin(phi),np.cos(phi)]])
	else:
		R_x = np.array([[1,0,0],
						[0,np.cos(phi),-np.sin(phi)],
						[0,np.sin(phi), np.cos(phi)]])
	return R_x

def get_Ry(theta,transpose=False):
	""" Rotation matrix along y-axis """
	if transpose:
		R_y = np.array([[np.cos(theta),0,-np.sin(theta)],
						[0,1,0],
						[np.sin(theta),0, np.cos(theta)]])
	else:
		R_y = np.array([[ np.cos(theta),0,np.sin(theta)],
						[0,1,0],
						[-np.sin(theta),0,np.cos(theta)]])
	return R_y

def get_Rz(psi,transpose=False):
	""" Rotation matrix along z-axis """
	if transpose:
		R_z = np.array([[np.cos(psi), np.sin(psi),0],
						[-np.sin(psi),np.cos(psi),0],
						[0,0,1]])
	else:
		R_z = np.array([[np.cos(psi),-np.sin(psi),0],
						[np.sin(psi), np.cos(psi),0],
						[0,0,1]])
	return R_z

def get_dRxdt(phi, dphidt):
	""" Time derivative of R_x matrix """
	return dphidt*np.array([[0,0,0],
							[0,-np.sin(phi),-np.cos(phi)],
							[0,np.cos(phi),-np.sin(phi)]])

def get_dRydt(theta, dthetadt):
	""" Time derivative of R_y matrix """
	return dthetadt*np.array([[-np.sin(theta),0,np.cos(theta)],
							[0,0,0],
							[-np.cos(theta),0,-np.sin(theta)]])

def get_dRzdt(psi, dpsidt):
	""" Time derivative of R_z matrix """
	return dpsidt*np.array([[-np.sin(psi),-np.cos(psi),0],
							[np.cos(psi),-np.sin(psi),0],
							[0,0,0]])

def get_dRdt(phi,theta,psi,dphidt,dthetadt,dpsidt):
	""" Computes the time derivative of the rotation matrix R """
	R_x = get_Rx(phi)
	R_y = get_Ry(theta)
	R_z = get_Rz(psi)
	dRx_dt = get_dRxdt(phi,dphidt)
	dRy_dt = get_dRydt(theta,dthetadt)
	dRz_dt = get_dRzdt(psi,dpsidt)
	Term1 = np.dot(np.dot(dRz_dt,R_y),R_x)
	Term2 = np.dot(np.dot(R_z,dRy_dt),R_x)
	Term3 = np.dot(np.dot(R_z,R_y),dRx_dt)
	return Term1 + Term2 + Term3

def rotation_matrix(phi,theta,psi,transpose=False):
	""" 
	Passive rotation matrix
	Phi rotates around x-axis
	Theta rotates around y-axis
	Psi rotates around z-axis
	"""
	"""
	R_x = get_Rx(phi,transpose=transpose)
	R_y = get_Ry(theta,transpose=transpose)
	R_z = get_Rz(psi,transpose=transpose)
	if transpose:
		R_xy = R_x.dot(R_y)
		R = R_xy.dot(R_z)
	else:
		R_zy = R_z.dot(R_y)
		R = R_zy.dot(R_x)
	"""
	cosphi = np.cos(phi)
	sinphi = np.sin(phi)
	costheta = np.cos(theta)
	sintheta = np.sin(theta)
	cospsi = np.cos(psi)
	sinpsi = np.sin(psi)
	cospsisintheta = cospsi*sintheta
	sinpsisintheta = sinpsi*sintheta
	R = np.zeros((3,3))
	R[0,0] = cospsi*costheta
	R[1,0] = sinpsi*costheta
	R[2,0] = -sintheta
	R[0,1] = -sinpsi*cosphi + cospsisintheta*sinphi
	R[1,1] = cospsi*cosphi + sinpsisintheta*sinphi
	R[2,1] = costheta*sinphi
	R[0,2] = sinpsi*sinphi + cospsisintheta*cosphi
	R[1,2] = -cospsi*sinphi + sinpsisintheta*cosphi
	R[2,2] = costheta*cosphi
	return R

def rotation_matrix_euler(e0, e1, e2, e3, transpose=False):
	R_matrix = np.zeros((3,3))
	R_matrix[0][0] = 2*(e0*e0 + e1*e1 - 0.5)
	R_matrix[0][1] = 2*(e1*e2 - e0*e3)
	R_matrix[0][2] = 2*(e1*e3 + e0*e2)
	
	R_matrix[1][0] = 2*(e1*e2 + e0*e3)
	R_matrix[1][1] = 2*(e0*e0 + e2*e2 - 0.5)
	R_matrix[1][2] = 2*(e2*e3 - e0*e1)

	R_matrix[2][0] = 2*(e1*e3 - e0*e2)
	R_matrix[2][1] = 2*(e2*e3 + e0*e1)
	R_matrix[2][2] = 2*(e0*e0 + e3*e3 - 0.5)
	return R_matrix

def moment_inertia_spheroid(a,c,rho,i):
	""" Values of the moment of inertia for spheroids """
	M = (4.0/15.0)*np.pi*rho*(a**2)*c
	if i == 1:
		return M*(a**2+c**2)
	elif i == 2:
		return M*(a**2+c**2)
	elif i == 3:
		return 2.0*M*a**2
	else:
		raise ValueError("Index i for moment_inertia_spheroid() not set properly.")

def moment_inertia_ellipsoid(a, b, c, rho):
	""" 
	Values of the moment of inertia for ellipsoids
	"""
	M = (4.0/15.0)*np.pi*rho*a*b*c
	I = np.zeros((3,3))
	I[0][0] = M*(b*b+c*c)
	I[1][1] = M*(a*a+c*c)
	I[2][2] = M*(b*b+a*a)
	return I

def angular_acceleration_ode(a,b,c,rho,p1,p2,torq,component):
	""" 
	The right hand side of the differential equation of p,q,r.
	See John's document 'rotational motion' for more info.
	"""
	I_11 = moment_inertia_spheroid(a,c,rho,1)
	I_22 = moment_inertia_spheroid(a,c,rho,2)
	I_33 = moment_inertia_spheroid(a,c,rho,3)
	if component == 1:	# Computes dp/dt
		return (torq - (I_33 - I_22)*p1*p2)/I_11
	elif component == 2:	# Computes dq/dt
		return (torq - (I_11 - I_33)*p1*p2)/I_22
	elif component == 3:	# Computes dr/dt
		return (torq - (I_22 - I_11)*p1*p2)/I_33
	else:
		raise ValueError("component not set right for angular_acceleration_ode().")

def angular_speed_ode(p, q, r, phi, theta):
	""" ODE for the angular speeds of p,q,r """
	sinp = np.sin(phi)
	cosp = np.cos(phi)
	dphidt = p + (sinp*q + cosp*r)*np.tan(theta)
	dthetadt = q*cosp - r*sinp
	dpsidt = (q*sinp + r*cosp)/np.cos(theta)
	return dphidt, dthetadt, dpsidt

def Rotation_matrix_components_new(phi, theta, psi, transpose, trigger):
	sinphi = np.sin(phi)
	cosphi = np.cos(phi)
	sintheta = np.sin(theta)
	costheta = np.cos(theta)
	sinpsi = np.sin(psi)
	cospsi = np.cos(psi)
	R_matrix = np.zeros((3,3))
	if (trigger == 0):
		# For Z-Y-X rotation -- original
		if (transpose == 0):
			sinpsint = sinphi*sintheta
			cospsint = cosphi*sintheta
			R_matrix[0][0] = cospsi*costheta
			R_matrix[1][0] = sinpsi*costheta
			R_matrix[2][0] = -sintheta
			R_matrix[0][1] = -sinpsi*cosphi + cospsi*sinpsint
			R_matrix[1][1] = cospsi*cosphi + sinpsi*sinpsint
			R_matrix[2][1] = costheta*sinphi
			R_matrix[0][2] = sinpsi*sinphi + cospsi*cospsint
			R_matrix[1][2] = -cospsi*sinphi + sinpsi*cospsint
			R_matrix[2][2] = costheta*cosphi
		else:
			cospsint = cospsi*sintheta
			sinpsint = sinpsi*sintheta
			R_matrix[0][0] = cospsi*costheta
			R_matrix[1][0] = -sinpsi*cosphi + cospsint*sinphi
			R_matrix[2][0] = sinpsi*sinphi + cospsint*cosphi
			R_matrix[0][1] = sinpsi*costheta
			R_matrix[1][1] = cospsi*cosphi + sinpsint*sinphi
			R_matrix[2][1] = -cospsi*sinphi + sinpsint*cosphi
			R_matrix[0][2] = -sintheta
			R_matrix[1][2] = costheta*sinphi
			R_matrix[2][2] = costheta*cosphi
	elif (trigger == 1):
		# For X-Z-Y rotation
		if (transpose == 0):
			R_matrix[0][0] = cospsi*costheta
			R_matrix[1][0] = cosphi*sinpsi*costheta + sinphi*sintheta
			R_matrix[2][0] = sinphi*sinpsi*costheta - cosphi*sintheta
			R_matrix[0][1] = -sinpsi
			R_matrix[1][1] = cosphi*cospsi
			R_matrix[2][1] = sinphi*cospsi
			R_matrix[0][2] = cospsi*sintheta
			R_matrix[1][2] = cosphi*sinpsi*sintheta - sinphi*costheta
			R_matrix[2][2] = sinphi*sinpsi*sintheta + cosphi*costheta
		else:
			R_matrix[0][0] = cospsi*costheta
			R_matrix[0][1] = cosphi*sinpsi*costheta + sinphi*sintheta
			R_matrix[0][2] = sinphi*sinpsi*costheta - cosphi*sintheta
			R_matrix[1][0] = -sinpsi
			R_matrix[1][1] = cosphi*cospsi
			R_matrix[1][2] = sinphi*cospsi
			R_matrix[2][0] = cospsi*sintheta
			R_matrix[2][1] = cosphi*sinpsi*sintheta - sinphi*costheta
			R_matrix[2][2] = sinphi*sinpsi*sintheta + cosphi*costheta
	elif (trigger == 2):
		# For Y-X-Z rotation
		if (transpose == 0):
			R_matrix[0][0] = costheta*cospsi + sintheta*sinphi*sinpsi
			R_matrix[1][0] = cosphi*sinpsi
			R_matrix[2][0] = -sintheta*cospsi + costheta*sinphi*sinpsi
			R_matrix[0][1] = -costheta*sinpsi + sintheta*sinphi*cospsi
			R_matrix[1][1] = cosphi*cospsi
			R_matrix[2][1] = sintheta*sinpsi + costheta*sinphi*cospsi
			R_matrix[0][2] = sintheta*cosphi
			R_matrix[1][2] = -sinphi
			R_matrix[2][2] = costheta*cosphi
		else:
			R_matrix[0][0] = costheta*cospsi + sintheta*sinphi*sinpsi
			R_matrix[0][1] = cosphi*sinpsi
			R_matrix[0][2] = -sintheta*cospsi + costheta*sinphi*sinpsi
			R_matrix[1][0] = -costheta*sinpsi + sintheta*sinphi*cospsi
			R_matrix[1][1] = cosphi*cospsi
			R_matrix[1][2] = sintheta*sinpsi + costheta*sinphi*cospsi
			R_matrix[2][0] = sintheta*cosphi
			R_matrix[2][1] = -sinphi
			R_matrix[2][2] = costheta*cosphi
	else:
		raise ValueError("Wrong trigger input. ")
	return R_matrix


def normal_vector_face_rh(indices, v4, vertices):
	"""
	Returns vertex indices in a right-handed ordering. 
	Also returns corresponding normal vector.
	"""
	Setup = True
	#ids = indices
	v1, v2, v3 = indices
	ids = np.array([v1, v2, v3], dtype=np.int32)
	while Setup:
		v1, v2, v3 = ids
		r1 = vertices[v1]
		r2 = vertices[v2]
		r3 = vertices[v3]
		r4 = vertices[v4]

		r2r1 = r2 - r1
		r3r1 = r3 - r1
		r4r1 = r4 - r1
		cross_prod = np.cross(r3r1, r2r1)
		norm_cproduct = np.linalg.norm(cross_prod)
		n_i = cross_prod#/norm_cproduct
		d = np.dot(n_i, r4r1)
		#print(n_i, d, ids)
		if d <= 0:
			Setup = False
		else:
			ids = np.flip(ids, 0)
	indices[0] = ids[0]
	indices[1] = ids[1]
	indices[2] = ids[2]
	return n_i

def normal_vector_face_summed(v1, v2, v3, v4, vertices):
	r1 = vertices[v1]
	r2 = vertices[v2]
	r3 = vertices[v3]
	r4 = vertices[v4]

	r2r1 = r2 - r1
	r3r1 = r3 - r1
	r4r1 = r4 - r1
	cross_prod = np.cross(r3r1, r2r1)
	norm_cproduct = np.linalg.norm(cross_prod)
	n_i = cross_prod#/norm_cproduct
	d = np.dot(n_i, r4r1)
	if d < 0:
		return n_i
	else:
		return -n_i

def get_polyhedron_data(vertices, N_vertices):
	"""
	Computes centroid and volume of a polyhedron.
	Also returns vertex indices that forms an applicable face of a polyhedron.
	Ignores faces that intersect the polyhedron.
	"""
	M = N_vertices-3
	n_vectors = np.zeros((M, 3))
	dist = np.zeros(int((M-1)*M/2))	
	N_faces = 0
	N_false = 0
	N_true = 0
	Volume = 0
	centroid = np.zeros(3)
	index_combo = np.zeros(0, dtype=np.int32)
	for i in range(N_vertices):
		for j in range(N_vertices):
			for k in range(N_vertices):
				if i < j and j < k:
					counter = 0
					for l in range(N_vertices):
						indices_in = np.array([i, j, k])
						indices = np.array([i+1, j+1, k+1, l+1], dtype=np.int32)
						if len(np.unique(indices)) == len(indices):
							##n_vectors[counter] = normal_vector_face_summed(i, j, k, l, vertices)
							n_vectors[counter] = normal_vector_face_rh(indices_in, l, vertices)
							counter += 1

					counter = 0
					if len(dist) > 0:
						for m in range(len(n_vectors)):
							for n in range(len(n_vectors)-1):
								if m > n:
									dist[counter] = np.dot(n_vectors[m], n_vectors[n])
									counter += 1
						if (dist > 0).all():
							A,B,C = indices_in
							N_faces += 1
							centroid[0] += np.dot(n_vectors[0],np.array([1,0,0]))*(np.dot(vertices[A]+vertices[B],np.array([1,0,0]))**2 \
										 + np.dot(vertices[B]+vertices[C],np.array([1,0,0]))**2 \
										 + np.dot(vertices[A]+vertices[C],np.array([1,0,0]))**2)/24.0
							centroid[1] += np.dot(n_vectors[0],np.array([0,1,0]))*(np.dot(vertices[A]+vertices[B],np.array([0,1,0]))**2 \
										 + np.dot(vertices[B]+vertices[C],np.array([0,1,0]))**2 \
										 + np.dot(vertices[A]+vertices[C],np.array([0,1,0]))**2)/24.0
							centroid[2] += np.dot(n_vectors[0],np.array([0,0,1]))*(np.dot(vertices[A]+vertices[B],np.array([0,0,1]))**2 \
										 + np.dot(vertices[B]+vertices[C],np.array([0,0,1]))**2 \
										 + np.dot(vertices[A]+vertices[C],np.array([0,0,1]))**2)/24.0
							Volume += np.dot(vertices[A], n_vectors[0])/6.0
							index_combo = np.concatenate([index_combo, np.array([A,B,C])])
							N_true += 1
						else:
							N_false += 1
					else:
						A,B,C = indices_in
						N_faces += 1
						Volume += np.dot(vertices[i], n_vectors[0])/6.0
						centroid[0] += np.dot(n_vectors[0],np.array([1,0,0]))*(np.dot(vertices[A]+vertices[B],np.array([1,0,0]))**2 \
									 + np.dot(vertices[B]+vertices[C],np.array([1,0,0]))**2 \
									 + np.dot(vertices[A]+vertices[C],np.array([1,0,0]))**2)/24.0
						centroid[1] += np.dot(n_vectors[0],np.array([0,1,0]))*(np.dot(vertices[A]+vertices[B],np.array([0,1,0]))**2 \
									 + np.dot(vertices[B]+vertices[C],np.array([0,1,0]))**2 \
									 + np.dot(vertices[A]+vertices[C],np.array([0,1,0]))**2)/24.0
						centroid[2] += np.dot(n_vectors[0],np.array([0,0,1]))*(np.dot(vertices[A]+vertices[B],np.array([0,0,1]))**2 \
									 + np.dot(vertices[B]+vertices[C],np.array([0,0,1]))**2 \
									 + np.dot(vertices[A]+vertices[C],np.array([0,0,1]))**2)/(24.0)
						index_combo = np.concatenate([index_combo, np.array([A,B,C])])
	centroid /= 2*Volume
	return centroid, index_combo, Volume, N_faces

def polyhedron_volume_centroid(vertices, semiaxes, p_info, face_ids):
	"""
	Computes the volume of the polyhedron and its centroid.
	Moves the vertices so they are centered around the centroid.
	Also computes the volume of an ellipsoid if applicable.
	"""
	
	NumBodies = int(len(p_info)/2)
	total_vertices = np.sum(p_info[:NumBodies])
	vertices_3d = vertices.reshape((total_vertices, 3))
	faces_iterated = 0
	id_fac = 0
	body_volumes = np.zeros(NumBodies)
	ad = np.zeros(3)
	bd = np.zeros(3)
	cd = np.zeros(3)
	for i in range(NumBodies):
		centroid = np.zeros(3)
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
			if num_faces == 4:
				# Volume for tetrahedron. Known
				for j in range(3):
					ad[j] = vertices[j + id_fac] - vertices[9+j + id_fac]
					bd[j] = vertices[3+j + id_fac] - vertices[9+j + id_fac]
					cd[j] = vertices[6+j + id_fac] - vertices[9+j + id_fac]
				cprod = np.cross(bd, cd)
				volume = np.abs(np.dot(ad, cprod))/6
				for j in range(4):
					centroid[0] += vertices[3*j + id_fac]/4
					centroid[1] += vertices[3*j+1 + id_fac]/4
					centroid[2] += vertices[3*j+2 + id_fac]/4
			else:
				vertices_in = np.zeros(nvertices*3)
				for j in range(3*nvertices):
					vertices_in[j] = vertices[id_fac + j]
				v_in = vertices_in.reshape((nvertices,3))
				# For other general polygon shapes
				volume = 0
				for j in range(num_faces):
					v1 = face_ids_in[3*j]
					v2 = face_ids_in[3*j+1]
					v3 = face_ids_in[3*j+2]
					for k in range(nvertices):
						if k != v1 and k != v2 and k != v3:
							n_i = normal_vector_face_summed(v1, v2, v3, k, v_in)
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
			vertices_3d -= centroid
			id_fac += 3*nvertices
		else:
			# For ellipsoid
			volume = (4.0/3.0)*np.pi*semiaxes[i][0]*semiaxes[i][1]*semiaxes[i][2]
		body_volumes[i] = volume
	return vertices_3d.ravel(), body_volumes


def polyhedron_volume_centroid_single(vertices, nvertices, num_faces, face_ids):
	"""
	Computes the volume of the polyhedron and its centroid.
	Moves the vertices so they are centered around the centroid.

	Algorithm as follows:
	1) Consider a face that is a triangle. Create a tetrahedron with a fourth point that is the centroid.
	2) Compute the volume of said tetrahedron.
	3) Sum the volume of all the tetrahedra in the polyhedron.
	"""
	vertices_3d = vertices.reshape((nvertices, 3))
	face_ids_3d = face_ids.reshape((num_faces, 3))
	
	centroid = np.zeros(3)
	centroid[0] = np.sum(vertices_3d[:,0])/nvertices
	centroid[1] = np.sum(vertices_3d[:,1])/nvertices
	centroid[2] = np.sum(vertices_3d[:,2])/nvertices
	vertices_3d -= centroid
	if num_faces == 4:
		# Volume for tetrahedron. Known
		v1 = vertices_3d[0]
		v2 = vertices_3d[1]
		v3 = vertices_3d[2]
		v4 = vertices_3d[3]
		# Volume of tetrahedron
		ad = v1 - v4
		bd = v2 - v4
		cd = v3 - v4
		cprod = np.cross(bd, cd)
		volume = np.abs(np.dot(ad, cprod))/6
	else:
		# Algorithm follows that of Dobrovolskis (1996)
		volume = 0
		for i in range(num_faces):
			fid1, fid2, fid3 = face_ids_3d[i]
			v1 = vertices_3d[fid1]
			v2 = vertices_3d[fid2]
			v3 = vertices_3d[fid3]
			for k in range(nvertices):
				if k != fid1 and k != fid2 and k != fid3:
					n_i = normal_vector_face_summed(fid1, fid2, fid3, k, vertices_3d)
					break
			volume += np.abs(np.dot(v1, n_i))/6			
	return vertices_3d.ravel(), volume


def rot_matrix_eulerparam_4x4(e0, e1, e2, e3):
	R = 2*np.array([[e0**2 + e1**2 - 0.5, e1*e2 - e0*e3, e1*e3 + e0*e2, 0],
					[e1*e2 + e0*e3, e0**2 + e2**2 - 0.5, e2*e3 - e0*e1, 0],
					[e1*e3 - e0*e2, e2*e3 + e0*e1, e0**2 + e3**2 - 0.5, 0],
					[0 ,0, 0, 0.5]])
	return R

def S_matrix(a, b, c, x0, y0, z0):
	A = 1/a**2
	B = 1/b**2
	C = 1/c**2
	G = -2*x0/a**2
	H = -2*y0/b**2
	J = -2*z0/c**2
	K = (x0/a)**2 + (y0/b)**2 + (z0/c)**2 - 1
	S = 0.5*np.array([[2*A, 0, 0, G],
					  [0, 2*B, 0, H],
					  [0, 0, 2*C, J],
					  [G, H, J, 2*K]])
		
	return S

def S_matrix2(a, b, c, x0, y0, z0, phi, theta, psi, e3, euler=False):
	A = 1/a**2
	B = 1/b**2
	C = 1/c**2
	K = -1
	S = 0.5*np.array([[2*A, 0, 0, 0],
					  [0, 2*B, 0, 0],
					  [0, 0, 2*C, 0],
					  [0, 0, 0, 2*K]])
	if euler:
		R = rot_matrix_eulerparam_4x4(phi, theta, psi, e3)
	else:
		R = rot_matrix_4x4(phi, theta, psi)
	newpos = np.dot(np.array([x0, y0, z0, 1]),R)
	T = np.array([[1, 0, 0, 0],
				  [0, 1, 0, 0],
				  [0, 0, 1, 0],
				  [-newpos[0], -newpos[1], -newpos[2], 1]])
	#S_return = np.dot(np.dot(T, S), T.transpose())
	S_return = S_matrix(a,b,c,newpos[0],newpos[1],newpos[2])
	S_return2 = np.dot(np.dot(R, S_return), R.transpose())
	return S_return2

def rot_matrix_4x4(phi, theta, psi):
	Rx = np.array([[1, 0, 0, 0],
				   [0, np.cos(phi), -np.sin(phi), 0],
				   [0, np.sin(phi), np.cos(phi), 0],
				   [0, 0, 0, 1]])
	Ry = np.array([[np.cos(theta), 0, np.sin(theta), 0],
				   [0, 1, 0, 0],
				   [-np.sin(theta), 0, np.cos(theta), 0],
				   [0, 0, 0, 1]])
	Rz = np.array([[np.cos(psi), -np.sin(psi), 0, 0],
				   [np.sin(psi), np.cos(psi), 0, 0],
				   [0, 0, 1, 0],
				   [0, 0, 0, 1]])
	R = np.dot(np.dot(Rz, Ry),Rx)
	return R

def ellipsoid_intersection_point(semiaxes, r1, r2, angles1, angles2, euler=False):
	a1 = semiaxes[0][0]
	b1 = semiaxes[0][1]
	c1 = semiaxes[0][2]
	a2 = semiaxes[1][0]
	b2 = semiaxes[1][1]
	c2 = semiaxes[1][2]
	phi1 = angles1[0]
	theta1 = angles1[1]
	psi1 = angles1[2]
	phi2 = angles2[0]
	theta2 = angles2[1]
	psi2 = angles2[2]

	#e0a, e1a, e2a, e3a = TB_to_euler_params(phi1, theta1, psi1)
	#e0b, e1b, e2b, e3b = TB_to_euler_params(phi2, theta2, psi2)
	if euler:
		A = S_matrix2(a1, b1, c1, r1[0], r1[1], r1[2], phi1, theta1, psi1, angles1[3], euler=True)
		B = S_matrix2(a2, b2, c2, r2[0], r2[1], r2[2], phi2, theta2, psi2, angles2[3], euler=True)
	else:
		A = S_matrix2(a1, b1, c1, r1[0], r1[1], r1[2], phi1, theta1, psi1, 0)
		B = S_matrix2(a2, b2, c2, r2[0], r2[1], r2[2], phi2, theta2, psi2, 0)
	A_inv = np.linalg.inv(A)
	M = np.dot(A_inv,B)
	eigval = eig(M, left=False, right=False)
	intersect = 0
	for i in range(2):
		if np.abs(eigval[i].imag) > 0:
			intersect = 1
			break
	
	if np.abs(eigval[0]-eigval[1]) <= 1e-15:
		intersect = 1

	intersect_pt = np.zeros(3)
	if intersect:
		for i in range(2):
			lhs = np.array([[M[0][0]-eigval[i], M[0][1], M[0][2]],
							[M[1][0], M[1][1]-eigval[i], M[1][2]],
							[M[2][0], M[2][1], M[2][2]-eigval[i]]])
			rhs = np.array([-M[0][3],-M[1][3],-M[2][3]])

			xv = np.linalg.solve(lhs, rhs)
			# Eigenvector
			x = np.concatenate([xv, [1]])
			intersect_pt = xv.real
			eq_sat = np.dot(np.dot(x,eigval[i]*A - B),x)
			if eq_sat < 1e-15:
				break
	return intersect_pt

def grav_field_spheroid(x, y, z, a, c, rho):
	""" Gravitational potential of a spheroid with a = b > c """
	x_sq = x*x
	y_sq = y*y
	z_sq = z*z
	a_sq = a*a
	c_sq = c*c
	B = a_sq + c_sq - x_sq - y_sq - z_sq
	C = a_sq*(c_sq - z_sq) - c_sq*(x_sq + y_sq)
	kappa = 0.5*(-B + np.sqrt(B*B - 4*C))
	prefac = np.pi*rho*a_sq*c
	if a > c:
		nasty_arcsin = np.arcsin(np.sqrt((a_sq-c_sq)/(a_sq+kappa)))/np.sqrt(a_sq - c_sq)
		g_prefac = prefac/(a_sq - c_sq)
		gx = 2*g_prefac*x*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gy = 2*g_prefac*y*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gz = 4*g_prefac*z*(nasty_arcsin - 1/np.sqrt(c_sq + kappa))	

	else:
		nasty_arcsinh = np.arcsinh(np.sqrt((a_sq-c_sq)/(a_sq+kappa)))/np.sqrt(c_sq - a_sq)
		g_prefac = prefac/(c_sq - a_sq)
		gx = 2*g_prefac*x*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gy = 2*g_prefac*y*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gz = 4*g_prefac*z*(1/np.sqrt(c_sq + kappa) - nasty_arcsinh)
		
	return gx, gy, gz

def grav_field_ellipsoid(x, y, z, a, b, c, rho):
	""" 
	Gravitational field of an ellipsoid with a > b > c 
	Calls for spheroid if a = b
	"""
	if (a-b) < 1e-10:
		gx, gy, gz = grav_field_spheroid(x, y, z, a, c, rho)
	else:
		x_sq = x*x
		y_sq = y*y
		z_sq = z*z
		a_sq = a*a
		b_sq = b*b
		c_sq = c*c

		ab_sq = a_sq*b_sq;
		ac_sq = a_sq*c_sq;
		bc_sq = b_sq*c_sq;
		
		B_coef = a_sq + b_sq + c_sq - (x_sq + y_sq + z_sq);
		C_coef = ab_sq + ac_sq + bc_sq - x_sq*(b_sq + c_sq) - y_sq*(a_sq + c_sq) - z_sq*(a_sq + b_sq);
		D_coef = ab_sq*c_sq - x_sq*bc_sq - y_sq*ac_sq - z_sq*ab_sq;

		coeffs = [1, B_coef, C_coef, D_coef]
		kappa = np.max(np.roots(coeffs))
		k = np.sqrt((a_sq - b_sq)/(a_sq - c_sq))
		omega_kappa = np.arcsin(np.sqrt((a_sq-c_sq)/(a_sq + kappa)))
		F_eliptc = ellipkinc(omega_kappa, k**2)
		E_eliptc = ellipeinc(omega_kappa, k**2)
		nasty_square_roots = np.sqrt(a_sq - c_sq)/np.sqrt((a_sq+kappa)*(b_sq+kappa)*(c_sq+kappa))
		prefac = 2*np.pi*rho*a*b*c/np.sqrt(a_sq-c_sq)
		gx = 2*x*prefac*(E_eliptc - F_eliptc)/(a_sq - b_sq)
		gy = 2*y*prefac*(F_eliptc/(a_sq-b_sq) - (a_sq-c_sq)*E_eliptc/((a_sq-b_sq)*(b_sq-c_sq)) 
					+ ((c_sq+kappa)/(b_sq-c_sq))*nasty_square_roots)
		gz = 2*z*prefac*(E_eliptc/(b_sq-c_sq) - ((b_sq+kappa)/(b_sq-c_sq))*nasty_square_roots)
	return gx, gy, gz
		
def spherical_harmonics_potential_2nd(mu, rvec, r, semiaxes, angles, use_euler_param):
	""" 
	Spherical harmonics coefficients up to second order 
	Expressions based on Boldrin et. al. (2016)
	Assumes equatorial radius R = a, largest semiaxis
	"""
	a, b, c = semiaxes
	if use_euler_param:
		R = rotation_matrix_euler(angles[0], angles[1], angles[2], angles[3])
	else:
		R = rotation_matrix(angles[0], angles[1], angles[2])
	R_rad = c
	C_20 = -(c**2 - (a**2 + b**2)/2)/(5*R_rad**2)
	C_22 = (a**2 - b**2)/(20*R_rad**2)
	xp, yp, zp = np.dot(R, rvec)
	U_2 = (mu/r**3)*(C_20*(3*zp**2/r**2 - 1)/2 + 3*C_22*(xp**2 - yp**2)/r**2)
	return U_2

def potential_ellipsoid_and_gravfield(x, y, z, a, b, c, rho):
	x_sq = x*x
	y_sq = y*y
	z_sq = z*z
	a_sq = a*a
	b_sq = b*b
	c_sq = c*c

	ab_sq = a_sq*b_sq;
	ac_sq = a_sq*c_sq;
	bc_sq = b_sq*c_sq;
	
	B_coef = a_sq + b_sq + c_sq - (x_sq + y_sq + z_sq);
	C_coef = ab_sq + ac_sq + bc_sq - x_sq*(b_sq + c_sq) - y_sq*(a_sq + c_sq) - z_sq*(a_sq + b_sq);
	D_coef = ab_sq*c_sq - x_sq*bc_sq - y_sq*ac_sq - z_sq*ab_sq;

	coeffs = [1, B_coef, C_coef, D_coef]
	kappa = np.max(np.roots(coeffs))
	k = (a_sq - b_sq)/(a_sq - c_sq)  # Ignore square root for scipy elliptic function
	omega_kappa = np.arcsin(np.sqrt((a_sq-c_sq)/(a_sq + kappa)))
	F_eliptc = ellipkinc(omega_kappa, k) # Scipy elliptic function uses k modulus and not k^2
	E_eliptc = ellipeinc(omega_kappa, k)
	nasty_square_roots = np.sqrt(a_sq - c_sq)/np.sqrt((a_sq+kappa)*(b_sq+kappa)*(c_sq+kappa))
	prefac = 2*np.pi*rho*a*b*c/np.sqrt(a_sq-c_sq)
	Term1 = (1 - x_sq/(a_sq - b_sq) + y_sq/(a_sq - b_sq))*F_eliptc
	Term2 = (x_sq/(a_sq - b_sq) - y_sq*(a_sq - c_sq)/((a_sq - b_sq)*(b_sq - c_sq)) + z_sq/(b_sq - c_sq))*E_eliptc
	Term3 = (y_sq*(c_sq + kappa)/(b_sq - c_sq) - z_sq*(b_sq + kappa)/(b_sq - c_sq))*nasty_square_roots
	Phi_potential = prefac*(Term1 + Term2 + Term3)
	"""
	print(Phi_potential)

	Phi_potential = prefac*((1 - x_sq/(a_sq-b_sq) + y_sq/(a_sq-b_sq))*F_eliptc
	+ (x_sq/(a_sq-b_sq) - ((a_sq-c_sq)*y_sq)/((a_sq-b_sq)*(b_sq-c_sq)) + z_sq/(b_sq-c_sq))*E_eliptc
	+ (y_sq*(c_sq + kappa)/(b_sq-c_sq) - z_sq*(b_sq + kappa)/(b_sq-c_sq))*(np.sqrt(a_sq-c_sq)/np.sqrt((a_sq+kappa)*(b_sq+kappa)*(c_sq+kappa))));
	print(Phi_potential)
	"""
	#dgx_dx, dgy_dx, dgz_dx = grav_field_ellipsoid_xterm_ddx(x, y, z, a, b, c, rho, E_eliptc, F_eliptc, kappa)
	#gx, gy, gz = grav_field_ellipsoid(x, y, z, a, b, c, rho, E_eliptc, F_eliptc, kappa)
	gx = 2*x*prefac*(E_eliptc - F_eliptc)/(a_sq - b_sq)
	gy = 2*y*prefac*(F_eliptc/(a_sq-b_sq) - (a_sq-c_sq)*E_eliptc/((a_sq-b_sq)*(b_sq-c_sq)) 
				+ ((c_sq+kappa)/(b_sq-c_sq))*nasty_square_roots)
	gz = 2*z*prefac*(E_eliptc/(b_sq-c_sq) - ((b_sq+kappa)/(b_sq-c_sq))*nasty_square_roots)
	"""
	#print(gx, gy, gz)
	gx = (2*x*prefac)*(E_eliptc - F_eliptc)/(a_sq-b_sq)
	gy = (2*y*prefac)*((F_eliptc/(a_sq-b_sq)) - (a_sq-c_sq)*E_eliptc/((a_sq-b_sq)*(b_sq-c_sq))
		+ ((c_sq+kappa)/(b_sq-c_sq))*(np.sqrt(a_sq-c_sq)/np.sqrt((a_sq+kappa)*(b_sq+kappa)*(c_sq+kappa))))
	gz = (2*z*prefac)*(E_eliptc/(b_sq-c_sq)
		- ((b_sq+kappa)/(b_sq-c_sq))*(np.sqrt(a_sq-c_sq)/np.sqrt((a_sq+kappa)*(b_sq+kappa)*(c_sq+kappa))))
	"""
	#print(gx, gy, gz)
	#sys.exit(0)

	gx_dx = 2*prefac*(E_eliptc - F_eliptc)/(a_sq - b_sq)
	gy_dy = 2*prefac*(F_eliptc/(a_sq-b_sq) - (a_sq-c_sq)*E_eliptc/((a_sq-b_sq)*(b_sq-c_sq)) 
				+ ((c_sq+kappa)/(b_sq-c_sq))*nasty_square_roots)
	gz_dz = 2*prefac*(E_eliptc/(b_sq-c_sq) - ((b_sq+kappa)/(b_sq-c_sq))*nasty_square_roots)
	return Phi_potential, gx, gy, gz, gx_dx, gy_dy, gz_dz

def potential_spheroid(x, y, z, a, c, rho):
	x_sq = x*x
	y_sq = y*y
	z_sq = z*z
	a_sq = a*a
	c_sq = c*c
	B = a_sq + c_sq - x_sq - y_sq - z_sq
	C = a_sq*(c_sq - z_sq) - c_sq*(x_sq + y_sq)
	kappa = 0.5*(-B + np.sqrt(B*B - 4*C))
	prefac = np.pi*rho*a_sq*c
	if a > c:
		nasty_arcsin = np.arcsin(np.sqrt((a_sq-c_sq)/(a_sq+kappa)))/np.sqrt(a_sq - c_sq)
		T1 = 2*prefac*(1 - (x_sq + y_sq - 2*z_sq)/(2*(a_sq - c_sq)))*nasty_arcsin
		T2 = prefac*np.sqrt(c_sq + kappa)*(x_sq + y_sq)/((a_sq - c_sq)*(a_sq + kappa))
		T3 = -2*prefac*z_sq/((a_sq - c_sq)*np.sqrt(c_sq + kappa))

		g_prefac = prefac/(a_sq - c_sq)
		gx = 2*g_prefac*x*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gy = 2*g_prefac*y*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gz = 4*g_prefac*z*(nasty_arcsin - 1/np.sqrt(c_sq + kappa))
		gx_dx = 2*g_prefac*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gy_dy = 2*g_prefac*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gz_dz = 4*g_prefac*(nasty_arcsin - 1/np.sqrt(c_sq + kappa))

	else:
		nasty_arcsinh = np.arcsinh(np.sqrt((a_sq-c_sq)/(a_sq+kappa)))/np.sqrt(c_sq - a_sq)
		T1 = 2*prefac*(1 + (x_sq + y_sq - 2*z_sq)/(2*(c_sq - a_sq)))*nasty_arcsinh
		T2 = -prefac*np.sqrt(c_sq + kappa)*(x_sq + y_sq)/((c_sq - a_sq)*(a_sq + kappa))
		T3 = 2*prefac*z_sq/((a_sq - c_sq)*np.sqrt(c_sq + kappa))
		g_prefac = prefac/(c_sq - a_sq)
		gx = 2*g_prefac*x*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gy = 2*g_prefac*y*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gz = 4*g_prefac*z*(1/np.sqrt(c_sq + kappa) - nasty_arcsinh)
		gx_dx = 2*g_prefac*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gy_dy = 2*g_prefac*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gz_dz = 4*g_prefac*(1/np.sqrt(c_sq + kappa) - nasty_arcsinh)
		
	Phi_potential = T1 + T2 + T3
	return Phi_potential, gx, gy, gz, gx_dx, gy_dy, gz_dz

def potential_spheroid_spherical_cord(r, theta, phi, a, c, rho):
	r_sq = r*r
	xr = np.sin(theta)*np.cos(phi)
	yr = np.sin(theta)*np.sin(phi)
	zr = np.cos(theta)
	xr_sq = xr*xr
	yr_sq = yr*yr
	zr_sq = zr*zr
	a_sq = a*a
	c_sq = c*c
	B = a_sq + c_sq - r_sq*xr_sq - r_sq*yr_sq - r_sq*zr_sq
	C = a_sq*(c_sq - r_sq*zr_sq) - r_sq*c_sq*(xr_sq + yr_sq)
	kappa = 0.5*(-B + np.sqrt(B*B - 4*C))
	prefac = np.pi*rho*a_sq*c
	if a > c:
		nasty_arcsin = np.arcsin(np.sqrt((a_sq-c_sq)/(a_sq+kappa)))/np.sqrt(a_sq - c_sq)
		T1 = 2*prefac*(1 - r_sq*(xr_sq + yr_sq - 2*zr_sq)/(2*(a_sq - c_sq)))*nasty_arcsin
		T2 = prefac*np.sqrt(c_sq + kappa)*r_sq*(xr_sq + yr_sq)/((a_sq - c_sq)*(a_sq + kappa))
		T3 = -2*prefac*r_sq*zr_sq/((a_sq - c_sq)*np.sqrt(c_sq + kappa))

		T1_dr = -2*prefac*r*((xr_sq + yr_sq - 2*zr_sq)/(a_sq - c_sq))*nasty_arcsin
		T2_dr = 2*prefac*np.sqrt(c_sq + kappa)*r*(xr_sq + yr_sq)/((a_sq - c_sq)*(a_sq + kappa))
		T3_dr = -4*prefac*r*zr_sq/((a_sq - c_sq)*np.sqrt(c_sq + kappa))
		
		g_prefac = prefac/(a_sq - c_sq)
		gx = 2*g_prefac*r*xr*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gy = 2*g_prefac*r*yr*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gz = 4*g_prefac*r*zr*(nasty_arcsin - 1/np.sqrt(c_sq + kappa))
		gx_dr = 2*g_prefac*xr*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gy_dr = 2*g_prefac*yr*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gz_dr = 4*g_prefac*zr*(nasty_arcsin - 1/np.sqrt(c_sq + kappa))
		
	else:
		sys.exit("Prolate spheroid not implemented yet")
		nasty_arcsinh = np.arcsinh(np.sqrt((a_sq-c_sq)/(a_sq+kappa)))/np.sqrt(c_sq - a_sq)
		T1 = 2*prefac*(1 - (x_sq + y_sq - 2*z_sq)/(2*(c_sq - a_sq)))*nasty_arcsinh
		T2 = -prefac*np.sqrt(c_sq + kappa)*(x_sq + y_sq)/((a_sq - c_sq)*(a_sq + kappa))
		T3 = 2*prefac*z_sq/((a_sq - c_sq)*np.sqrt(c_sq + kappa))
		g_prefac = prefac/(c_sq - a_sq)
		gx = 2*g_prefac*x*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gy = 2*g_prefac*y*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gz = 4*g_prefac*z*(1/np.sqrt(c_sq + kappa) - nasty_arcsinh)
		gx_dx = 2*g_prefac*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gy_dy = 2*g_prefac*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gz_dz = 4*g_prefac*(1/np.sqrt(c_sq + kappa) - nasty_arcsinh)
		
	Phi_potential = T1 + T2 + T3
	Phi_potential_dr = T1_dr + T2_dr + T3_dr
	return Phi_potential, gx, gy, gz, gx_dr, gy_dr, gz_dr, Phi_potential_dr

def potential_spheroid_cylindrical_cord(r, phi, z, a, c, rho):
	r_sq = r*r
	xr = np.cos(phi)
	yr = np.sin(phi)
	xr_sq = xr*xr
	yr_sq = yr*yr
	z_sq = z*z
	a_sq = a*a
	c_sq = c*c
	B = a_sq + c_sq - r_sq*xr_sq - r_sq*yr_sq - z_sq
	C = a_sq*(c_sq - z_sq) - r_sq*c_sq*(xr_sq + yr_sq)
	kappa = 0.5*(-B + np.sqrt(B*B - 4*C))
	prefac = np.pi*rho*a_sq*c
	if a > c:
		nasty_arcsin = np.arcsin(np.sqrt((a_sq-c_sq)/(a_sq+kappa)))/np.sqrt(a_sq - c_sq)
		T1 = 2*prefac*(1 - (r_sq*xr_sq + r_sq*yr_sq - 2*z_sq)/(2*(a_sq - c_sq)))*nasty_arcsin
		T2 = prefac*np.sqrt(c_sq + kappa)*r_sq*(xr_sq + yr_sq)/((a_sq - c_sq)*(a_sq + kappa))
		T3 = -2*prefac*z_sq/((a_sq - c_sq)*np.sqrt(c_sq + kappa))

		T1_dr = -2*prefac*r*((xr_sq + yr_sq)/(a_sq - c_sq))*nasty_arcsin
		T2_dr = 2*prefac*np.sqrt(c_sq + kappa)*r*(xr_sq + yr_sq)/((a_sq - c_sq)*(a_sq + kappa))
		T3_dr = 0
		
		g_prefac = prefac/(a_sq - c_sq)
		gx = 2*g_prefac*r*xr*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gy = 2*g_prefac*r*yr*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gz = 4*g_prefac*z*(nasty_arcsin - 1/np.sqrt(c_sq + kappa))
		gx_dr = 2*g_prefac*xr*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gy_dr = 2*g_prefac*yr*(np.sqrt(c_sq + kappa)/(a_sq + kappa) - nasty_arcsin)
		gz_dr = 0
		
	else:
		sys.exit("Prolate spheroid not implemented yet")
		nasty_arcsinh = np.arcsinh(np.sqrt((a_sq-c_sq)/(a_sq+kappa)))/np.sqrt(c_sq - a_sq)
		T1 = 2*prefac*(1 - (x_sq + y_sq - 2*z_sq)/(2*(c_sq - a_sq)))*nasty_arcsinh
		T2 = -prefac*np.sqrt(c_sq + kappa)*(x_sq + y_sq)/((a_sq - c_sq)*(a_sq + kappa))
		T3 = 2*prefac*z_sq/((a_sq - c_sq)*np.sqrt(c_sq + kappa))
		g_prefac = prefac/(c_sq - a_sq)
		gx = 2*g_prefac*x*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gy = 2*g_prefac*y*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gz = 4*g_prefac*z*(1/np.sqrt(c_sq + kappa) - nasty_arcsinh)
		gx_dx = 2*g_prefac*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gy_dy = 2*g_prefac*(nasty_arcsinh - np.sqrt(c_sq + kappa)/(a_sq + kappa))
		gz_dz = 4*g_prefac*(1/np.sqrt(c_sq + kappa) - nasty_arcsinh)
		
	Phi_potential = T1 + T2 + T3
	Phi_potential_dr = T1_dr + T2_dr + T3_dr
	return Phi_potential, gx, gy, gz, gx_dr, gy_dr, gz_dr, Phi_potential_dr


def potential_ellipsoid_derivative_rx(z, alpha, semiaxes, rvec, G_g, rho1, rho2, R1, R2):
	a1, b1, c1 = semiaxes[0]
	a2, b2, c2 = semiaxes[1]
	aprime = a1*np.sqrt(c1*c1 - z*z)/c1
	bprime = b1*np.sqrt(c1*c1 - z*z)/c1
	x = aprime*np.cos(alpha)
	y = bprime*np.sin(alpha)
	r_surface = np.array([x, y, z])
	xp, yp, zp = np.dot(R2.transpose(), np.dot(R1, r_surface)-rvec)

	#Phi_ellip, gx, gy, gz, gx_dx, gy_dy, gz_dz = potential_ellipsoid_and_gravfield(xp, yp, zp, a2, b2, c2, rho2)
	if a2 == b2:
		Phi_ellip, gx, gy, gz, gx_dx, gy_dy, gz_dz = potential_spheroid(xp, yp, zp, a2, c2, rho2)
	else:
		Phi_ellip, gx, gy, gz, gx_dx, gy_dy, gz_dz = potential_ellipsoid_and_gravfield(xp, yp, zp, a2, b2, c2, rho2)
	nx = bprime*np.cos(alpha)
	ny = aprime*np.sin(alpha)
	nz = a1*b1*z/c1**2
	ndS = np.array([nx, ny, nz])
	
	#third_term_factor = r*(xp - yp*yp/xp - zp*zp/xp)/xp
	#x_comp = r*(xp*gx + Phi_ellip)/xp - third_term_factor*gx - r**3*gx/(2*xp**2)
	#y_comp = r*(yp*gx - yp*Phi_ellip/xp)/xp - third_term_factor*gy + r**3*gy/(2*xp**2)
	#z_comp = r*(zp*gx - zp*Phi_ellip/xp)/xp - third_term_factor*gz + r**3*gz/(2*xp**2)
	#x_comp = r*(x*gx + Phi_ellip)/xp - r*gx - r*gx*(x**2 + y**2 + z**2)/(2*xp**2)
	#y_comp = r*y*gx/xp - r*gy 
	#z_comp = r*z*gx/xp - r*gz
	#x_comp = x*r*gx/xp - r*(x**2 +y**2 + z**2)*gx/(2*xp**2)
	#y_comp = y*r*gx/xp
	#z_comp = z*r*gx/xp
	#integrand = -G_g*rho1*np.dot(np.array([x_comp, y_comp, z_comp]), ndS)/3
	r = np.sqrt(xp*xp + yp*yp + zp*zp)
	"""
	# Derivative of r*Phi
	term1_x = r*(div_spec(Phi_ellip, xp) + gx + div_spec(xp*gy, yp) + div_spec(xp*gz, zp))
	term1_y = r*(div_spec(Phi_ellip, yp) + gy + div_spec(yp*gx, xp) + div_spec(yp*gz, zp))
	term1_z = r*(div_spec(Phi_ellip, zp) + gz + div_spec(zp*gx, xp) + div_spec(zp*gy, yp))
	# Derivative of |r|^2*g/2
	term2_x = -3*r*gx - r**3*div_spec(gx_dx, xp)/2
	term2_y = -3*r*gy - r**3*div_spec(gy_dy, yp)/2
	term2_z = -3*r*gz - r**3*div_spec(gz_dz, zp)/2	
	# Combined first term for product rule
	term1_derivative = (term1_x + term2_x)*nx + (term1_y + term2_y)*ny + (term1_z + term2_z)*nz
	# Derivative of ndS
	term3_x = -r*div_spec(b1*np.cos(alpha), c1*np.sqrt(c1*c1 - z*z))
	term3_y = -r*div_spec(a1*np.sin(alpha), c1*np.sqrt(c1*c1 - z*z))
	term3_z = div_spec(a1*b1, z*c1*c1)
	# Combined second term for product rule
	term2_derivative = (xp*Phi_ellip - 0.5*r**2*gx)*term3_x + (yp*Phi_ellip - 0.5*r**2*gy)*term3_y + (zp*Phi_ellip - 0.5*r**2*gz)*term3_z
	"""
	# Using spherical coordinates
	Theta = np.arctan2(np.sqrt(xp*xp + yp*yp), zp)
	Phi = np.arctan2(yp, xp)
	xr = np.sin(Theta)*np.cos(Phi)
	yr = np.sin(Theta)*np.sin(Phi)
	zr = np.cos(Theta)
	term1_x = (Phi_ellip + xp*gx)*xr + xp*gy*yr + xp*gz*zr
	term1_y = yp*gx*xr + (Phi_ellip + yp*gy)*yr + yp*gz*zr
	term1_z = zp*gx*xr + zp*gy*yr + (Phi_ellip + zp*gz)*zr

	t2_prefac = xp*xr + yp*yr + zp*zr
	term2_x = -gx*t2_prefac - 0.5*xr*gx_dx*r**2
	term2_y = -gy*t2_prefac - 0.5*yr*gy_dy*r**2
	term2_z = -gz*t2_prefac - 0.5*zr*gz_dz*r**2
	# Combined first term for product rule
	term1_derivative = (term1_x + term2_x)*nx + (term1_y + term2_y)*ny + (term1_z + term2_z)*nz
	# Derivative of ndS
	term3_x = -b1*z*np.cos(alpha)*zr/(c1*np.sqrt(c1*c1-z*z))
	term3_y = -a1*z*np.sin(alpha)*zr/(c1*np.sqrt(c1*c1-z*z))
	term3_z = zr*a1*b1/c1**2
	# Combined second term for product rule
	term2_derivative = (xp*Phi_ellip - 0.5*r**2*gx)*term3_x + (yp*Phi_ellip - 0.5*r**2*gy)*term3_y + (zp*Phi_ellip - 0.5*r**2*gz)*term3_z
	"""
	# Treats r outside function as constant
	dphidr = xr*gx + yr*gy + zr*gz
	term1_x = x*dphidr
	term1_y = y*dphidr
	term1_z = z*dphidr
	term2_x = -0.5*(x*x + y*y + z*z)*xr*gx_dx
	term2_y = -0.5*(x*x + y*y + z*z)*yr*gy_dy
	term2_z = -0.5*(x*x + y*y + z*z)*zr*gz_dz
	term1_derivative = (term1_x + term2_x)*nx + (term1_y + term2_y)*ny + (term1_z + term2_z)*nz
	"""
	
	integrand = G_g*rho1*(term1_derivative+term2_derivative)/3
	return integrand


def potential_ellipsoid_derivative_spherical(z, alpha, semiaxes, rvec, G_g, rho1, rho2, R1, R2):
	a1, b1, c1 = semiaxes[0]
	a2, b2, c2 = semiaxes[1]
	aprime = a1*np.sqrt(c1*c1 - z*z)/c1
	bprime = b1*np.sqrt(c1*c1 - z*z)/c1
	x = aprime*np.cos(alpha)
	y = bprime*np.sin(alpha)
	r_surface = np.array([x, y, z])
	xp, yp, zp = np.dot(R2.transpose(), np.dot(R1, r_surface)-rvec)

	# Using spherical coordinates
	r = np.sqrt(xp*xp + yp*yp + zp*zp)
	Theta = np.arctan2(np.sqrt(xp*xp + yp*yp), zp)
	Phi = np.arctan2(yp, xp)
	xr = np.sin(Theta)*np.cos(Phi)
	yr = np.sin(Theta)*np.sin(Phi)
	zr = np.cos(Theta)
	if a2 == b2:
		Phi_ellip, gx, gy, gz, gx_dr, gy_dr, gz_dr, Phi_ellip_dr = potential_spheroid_spherical_cord(r, Theta, Phi, a2, c2, rho2)
	else:
		Phi_ellip, gx, gy, gz, gx_dx, gy_dy, gz_dz = potential_ellipsoid_and_gravfield(xp, yp, zp, a2, b2, c2, rho2)
	nx = bprime*np.cos(alpha)
	ny = aprime*np.sin(alpha)
	nz = a1*b1*z/c1**2
	ndS = np.array([nx, ny, nz])
	rr = np.sqrt(x*x + y*y + z*z)	
	term1_x = xp*Phi_ellip_dr + Phi_ellip*xr
	term1_y = yp*Phi_ellip_dr + Phi_ellip*yr
	term1_z = zp*Phi_ellip_dr + Phi_ellip*zr
	term2_x = -r*gx - r*r*gx_dr/2.0
	term2_y = -r*gy - r*r*gy_dr/2.0
	term2_z = -r*gz - r*r*gz_dr/2.0

	"""
	# Treated differently for r
	term1_x = x*Phi_ellip_dr
	term1_y = y*Phi_ellip_dr
	term1_z = z*Phi_ellip_dr
	term2_x = -rr*rr*gx_dr/2
	term2_y = -rr*rr*gy_dr/2
	term2_z = -rr*rr*gz_dr/2
	"""
	
	term1_derivative = (term1_x + term2_x)*nx + (term1_y + term2_y)*ny + (term1_z + term2_z)*nz
	
	"""
	term3_x = xr*a1/b1 -b1*z*np.cos(alpha)*zr/(c1*np.sqrt(c1*c1-z*z))
	term3_y = yr*a1/b1 -a1*z*np.cos(alpha)*zr/(c1*np.sqrt(c1*c1-z*z))
	term3_z = a1*b1/c1**2
	"""
	term3_x = -b1*r*np.cos(alpha)*zr*zr/(c1*np.sqrt(c1*c1-z*z))
	term3_y = -a1*r*np.sin(alpha)*zr*zr/(c1*np.sqrt(c1*c1-z*z))
	term3_z = zr*a1*b1/c1**2
	# Combined second term for product rule
	term2_derivative = (xp*Phi_ellip - 0.5*r**2*gx)*term3_x + (yp*Phi_ellip - 0.5*r**2*gy)*term3_y + (zp*Phi_ellip - 0.5*r**2*gz)*term3_z
	integrand = G_g*rho1*(term1_derivative+0*term2_derivative)/3
	return integrand

def potential_ellipsoid_derivative_cylindrical(z, alpha, semiaxes, rvec, G_g, rho1, rho2, R1, R2):
	a1, b1, c1 = semiaxes[0]
	a2, b2, c2 = semiaxes[1]
	aprime = a1*np.sqrt(c1*c1 - z*z)/c1
	bprime = b1*np.sqrt(c1*c1 - z*z)/c1
	x = aprime*np.cos(alpha)
	y = bprime*np.sin(alpha)
	r_surface = np.array([x, y, z])
	xp, yp, zp = np.dot(R2.transpose(), np.dot(R1, r_surface)-rvec)

	# Using spherical coordinates
	r = np.sqrt(xp*xp + yp*yp)
	if xp == 0 and yp == 0:
		Phi = 0
	elif xp >= 0:
		Phi = np.arctan2(yp,r)
	elif xp > 0:
		Phi = np.arctan2(yp,xp)
	else:
		Phi = -np.arcsin(yp/r) + np.pi
	xr = np.cos(Phi)
	yr = np.sin(Phi)
	if a2 == b2:
		Phi_ellip, gx, gy, gz, gx_dr, gy_dr, gz_dr, Phi_ellip_dr = potential_spheroid_cylindrical_cord(r, Phi, zp, a2, c2, rho2)
	else:
		Phi_ellip, gx, gy, gz, gx_dx, gy_dy, gz_dz = potential_ellipsoid_and_gravfield(xp, yp, zp, a2, b2, c2, rho2)
	nx = bprime*np.cos(alpha)
	ny = aprime*np.sin(alpha)
	nz = a1*b1*z/c1**2
	ndS = np.array([nx, ny, nz])
	rr = np.sqrt(x*x + y*y + z*z)	
	term1_x = xp*Phi_ellip_dr + Phi_ellip*xr
	term1_y = yp*Phi_ellip_dr + Phi_ellip*yr
	term1_z = zp*Phi_ellip_dr
	term2_x = -r*gx - r*r*gx_dr/2.0
	term2_y = -r*gy - r*r*gy_dr/2.0
	term2_z = -r*gz - r*r*gz_dr/2.0
	term1_derivative = (term1_x + term2_x)*nx + (term1_y + term2_y)*ny + (term1_z + term2_z)*nz

	"""
	r = np.sqrt(xp*xp + yp*yp + zp*zp)
	Theta = np.arctan2(np.sqrt(xp*xp + yp*yp), zp)
	Phi = np.arctan2(yp, xp)
	
	Phi_ellip2, gx2, gy2, gz2, gx_dr2, gy_dr2, gz_dr2, Phi_ellip_dr2 = potential_spheroid_spherical_cord(r, Theta, Phi, a2, c2, rho2)
	print(Phi_ellip - Phi_ellip2)
	print(gx - gx2)
	print(gy - gy2)
	print(gz - gz2)
	print(gx_dr - gx_dr2)
	print(gy_dr - gy_dr2)
	print("HI")
	print(gz_dr - gz_dr2)
	print(Phi_ellip_dr - Phi_ellip_dr2)
	sys.exit(0)
	
	term3_x = xr*a1/b1 -b1*z*np.cos(alpha)*zr/(c1*np.sqrt(c1*c1-z*z))
	term3_y = yr*a1/b1 -a1*z*np.cos(alpha)*zr/(c1*np.sqrt(c1*c1-z*z))
	term3_z = a1*b1/c1**2
	"""
	term3_x = -b1*r*np.cos(alpha)*z*z/(c1*np.sqrt(c1*c1-z*z))
	term3_y = -a1*r*np.sin(alpha)*z*z/(c1*np.sqrt(c1*c1-z*z))
	term3_z = z*a1*b1/c1**2
	# Combined second term for product rule
	term2_derivative = (xp*Phi_ellip - 0.5*r**2*gx)*term3_x + (yp*Phi_ellip - 0.5*r**2*gy)*term3_y + (zp*Phi_ellip - 0.5*r**2*gz)*term3_z
	integrand = G_g*rho1*(term1_derivative)/3
	return integrand

def potential_energy_conway(z, alpha, semiaxes, rvec, G_g, rho1, rho2, R1, R2):
	a1, b1, c1 = semiaxes[0]
	a2, b2, c2 = semiaxes[1]
	aprime = a1*np.sqrt(c1*c1 - z*z)/c1
	bprime = b1*np.sqrt(c1*c1 - z*z)/c1
	x = aprime*np.cos(alpha)
	y = bprime*np.sin(alpha)
	r_surface = np.array([x, y, z])
	xp, yp, zp = np.dot(R2.transpose(), np.dot(R1, r_surface)-rvec)
	#xp, yp, zp = r_surface - rvec
	r = np.sqrt(xp*xp + yp*yp + zp*zp)
	Theta = np.arctan2(np.sqrt(xp*xp + yp*yp), zp)
	Phi = np.arctan2(yp, xp)
	xr = np.sin(Theta)*np.cos(Phi)
	yr = np.sin(Theta)*np.sin(Phi)
	zr = np.cos(Theta)
	
	if a2 == b2:
		Phi_ellip, gx, gy, gz, gx_dx, gy_dy, gz_dz = potential_spheroid(xp, yp, zp, a2, c2, rho2)
		#Phi_ellip, gx, gy, gz, gx_dr, gy_dr, gz_dr, Phi_ellip_dr = potential_spheroid_spherical_cord(r, Theta, Phi, a2, c2, rho2)
	else:
		Phi_ellip, gx, gy, gz, gx_dx, gy_dy, gz_dz = potential_ellipsoid_and_gravfield(xp, yp, zp, a2, b2, c2, rho2)
	
	nx = bprime*np.cos(alpha)
	ny = aprime*np.sin(alpha)
	nz = a1*b1*z/c1**2
	ndS = np.array([nx, ny, nz])
	#x_comp = x*Phi_ellip - 0.5*(x*x + y*y + z*z)*gx
	#y_comp = y*Phi_ellip - 0.5*(x*x + y*y + z*z)*gy
	#z_comp = z*Phi_ellip - 0.5*(x*x + y*y + z*z)*gz
	#integrand = -G_g*rho1*np.dot(np.array([x_comp, y_comp, z_comp]), ndS)/3
	term1 = Phi_ellip*(xp*nx + yp*ny + zp*nz)
	term2 = 0.5*(xp*xp + yp*yp + zp*zp)*(gx*nx + gy*ny + gz*nz)
	integrand = G_g*rho1*(term1-term2)/3.0
	return integrand


def potential_ellipsoid(x, y, z, a, b, c, rho):
	x_sq = x*x
	y_sq = y*y
	z_sq = z*z
	a_sq = a*a
	b_sq = b*b
	c_sq = c*c

	ab_sq = a_sq*b_sq;
	ac_sq = a_sq*c_sq;
	bc_sq = b_sq*c_sq;
	
	B_coef = a_sq + b_sq + c_sq - (x_sq + y_sq + z_sq);
	C_coef = ab_sq + ac_sq + bc_sq - x_sq*(b_sq + c_sq) - y_sq*(a_sq + c_sq) - z_sq*(a_sq + b_sq);
	D_coef = ab_sq*c_sq - x_sq*bc_sq - y_sq*ac_sq - z_sq*ab_sq;

	coeffs = [1, B_coef, C_coef, D_coef]
	kappa = np.max(np.roots(coeffs))
	k = (a_sq - b_sq)/(a_sq - c_sq) # Ignore square root for scipy elliptic function
	omega_kappa = np.arcsin(np.sqrt((a_sq-c_sq)/(a_sq + kappa)))
	F_eliptc = ellipkinc(omega_kappa, k)  # Scipy elliptic function uses k modulus and not k^2
	E_eliptc = ellipeinc(omega_kappa, k)
	nasty_square_roots = np.sqrt(a_sq - c_sq)/np.sqrt((a_sq+kappa)*(b_sq+kappa)*(c_sq+kappa))
	prefac = 2*np.pi*rho*a*b*c/np.sqrt(a_sq-c_sq)
	Term1 = (1 - x_sq/(a_sq - b_sq) + y_sq/(a_sq - b_sq))*F_eliptc
	Term2 = (x_sq/(a_sq - b_sq) - y_sq*(a_sq - c_sq)/((a_sq - b_sq)*(b_sq - c_sq)) + z_sq/(b_sq - c_sq))*E_eliptc
	Term3 = (y_sq*(c_sq + kappa)/(b_sq - c_sq) - z_sq*(b_sq + kappa)/(b_sq - c_sq))*nasty_square_roots
	Phi_potential = prefac*(Term1 + Term2 + Term3)
	#print("%.5e, %.5e, %.5e, %.5e" %(k, omega_kappa, F_eliptc, E_eliptc))
	return Phi_potential
	
def force_conway(z, alpha, a1, b1, c1, a2, b2, c2, rvec, G_g, rho1, rho2, R1, R2, comp):
	aprime = a1*np.sqrt(c1*c1 - z*z)/c1
	bprime = b1*np.sqrt(c1*c1 - z*z)/c1
	x = aprime*np.cos(alpha)
	y = bprime*np.sin(alpha)
	r_surface = np.array([x, y, z])
	xp, yp, zp = np.dot(R2.transpose(), np.dot(R1, r_surface)-rvec)
	r = np.sqrt(xp*xp + yp*yp + zp*zp)
	Theta = np.arctan2(np.sqrt(xp*xp + yp*yp), zp)
	Phi = np.arctan2(yp, xp)
	xr = np.sin(Theta)*np.cos(Phi)
	yr = np.sin(Theta)*np.sin(Phi)
	zr = np.cos(Theta)
	if np.abs(a2 - b2) < 1e-13:
		Phi_ellip, gx, gy, gz, gx_dx, gy_dy, gz_dz = potential_spheroid(xp, yp, zp, a2, c2, rho2)
		#Phi_ellip, gx, gy, gz, gx_dr, gy_dr, gz_dr, Phi_ellip_dr = potential_spheroid_spherical_cord(r, Theta, Phi, a2, c2, rho2)
	else:
		#Phi_ellip, gx, gy, gz, gx_dx, gy_dy, gz_dz = potential_ellipsoid_and_gravfield(xp, yp, zp, a2, b2, c2, rho2)
		Phi_ellip = potential_ellipsoid(xp, yp, zp, a2, b2, c2, rho2)
	nx = bprime*np.cos(alpha)
	ny = aprime*np.sin(alpha)
	nz = a1*b1*z/c1**2
	#print(G_g*rho1*Phi_ellip, z, alpha)
	#print(rho1*rho2*G_g)
	#print(Phi_ellip, z, alpha)
	#sys.exit(0)
	if comp == 1:
		return G_g*rho1*Phi_ellip*nx
	elif comp == 2:
		return G_g*rho1*Phi_ellip*ny
	elif comp == 3:
		return G_g*rho1*Phi_ellip*nz

def omega_0_exact(m1, m2, r, F):
	""" 
	Computes the value of omega_0 by using exact expressions.
	Based on Scheeres 2009.
	"""
	x = r[0]
	y = r[1]
	z = r[2]
	# Spherical coordinates
	Theta = np.arctan2(np.sqrt(x*x + y*y), z)
	Phi = np.arctan2(y, x)
	
	dUdr = -np.sin(Theta)*np.cos(Phi)*F[0] - np.sin(Theta)*np.sin(Phi)*F[1] - np.cos(Theta)*F[2]
	omega_0_sq = (m1+m2)*np.abs(dUdr)/(m1*m2*np.linalg.norm(r))
	#print(np.sqrt(omega_0_sq))
	
	# Cylindrical coordinates
	r = np.sqrt(x*x + y*y)
	if x == 0 and y == 0:
		Phi = 0
	elif x >= 0:
		Phi = np.arctan2(y,r)
	elif x > 0:
		Phi = np.arctan2(y,x)
	else:
		Phi = -np.arcsin(y/r) + np.pi

	dUdr = -np.cos(Phi)*F[0] - np.sin(Phi)*F[1]
	omega_0_sq = (m1+m2)*np.abs(dUdr)/(m1*m2*np.linalg.norm(r))
	#print(np.sqrt(omega_0_sq))
	#sys.exit(0)
	return np.sqrt(omega_0_sq)

def omega_0_scheeres_exact(semiaxes, m1, m2, rho1, rho2, r, G_g, angles):
	R1 = rotation_matrix(angles[0], angles[1], angles[2])
	R2 = rotation_matrix(angles[3], angles[4], angles[5])
	a1 = semiaxes[0][0]
	b1 = semiaxes[0][1]
	c1 = semiaxes[0][2]
	a2 = semiaxes[1][0]
	b2 = semiaxes[1][1]
	c2 = semiaxes[1][2]
	semiaxes2 = np.array([semiaxes[1], semiaxes[0]])
	"""
	E = integrate.dblquad(potential_energy_conway, 0.0, 2*np.pi, -semiaxes[0][2], semiaxes[0][2], 
							args=(semiaxes, r, G_g, rho1, rho2, R1, R2))[0]
	E2 = integrate.dblquad(potential_energy_conway, 0.0, 2*np.pi, -semiaxes2[0][2], semiaxes2[0][2], 
							args=(semiaxes2, -r, G_g, rho1, rho2, R1, R2))[0]
	print(E)
	print(E2)
	print(np.abs(E-E2))
	sys.exit(0)
	#return E
	Fx1 = integrate.dblquad(force_conway, 0.0, 2*np.pi, -c1, c1, args=(a1, b1, c1, a2, b2, c2, r, G_g, rho1, rho2, R1, R2, 1))[0]
	Fy1 = integrate.dblquad(force_conway, 0.0, 2*np.pi, -c1, c1, args=(a1, b1, c1, a2, b2, c2, r, G_g, rho1, rho2, R1, R2, 2))[0]
	Fz1 = integrate.dblquad(force_conway, 0.0, 2*np.pi, -c1, c1, args=(a1, b1, c1, a2, b2, c2, r, G_g, rho1, rho2, R1, R2, 3))[0]
	Fx2 = integrate.dblquad(force_conway, 0.0, 2*np.pi, -c2, c2, args=(a2, b2, c2, a1, b1, c1, -r, G_g, rho1, rho2, R1, R2, 1))[0]
	Fy2 = integrate.dblquad(force_conway, 0.0, 2*np.pi, -c2, c2, args=(a2, b2, c2, a1, b1, c1, -r, G_g, rho1, rho2, R1, R2, 2))[0]
	Fz2 = integrate.dblquad(force_conway, 0.0, 2*np.pi, -c2, c2, args=(a2, b2, c2, a1, b1, c1, -r, G_g, rho1, rho2, R1, R2, 3))[0]
		
	F1 = np.array([Fx1, Fy1, Fz1])
	F2 = np.array([Fx2, Fy2, Fz2])
	print(F1)
	print(F2)
	print(np.abs(F2)-np.abs(F1))
	x = r[0]
	y = r[1]
	z = r[2]
	Theta = np.arctan2(np.sqrt(x*x + y*y), z)
	Phi = np.arctan2(y, x)
	dUdr = -np.sin(Theta)*np.cos(Phi)*F1[0] - np.sin(Theta)*np.sin(Phi)*F1[1] - np.cos(Theta)*F1[2]
	dUdr2 = -np.sin(Theta)*np.cos(Phi)*F2[0] - np.sin(Theta)*np.sin(Phi)*F2[1] - np.cos(Theta)*F2[2]
	"""
	#sys.exit(0)
	
	"""
	dUdr = integrate.dblquad(potential_ellipsoid_derivative_rx, 0, 2*np.pi, -semiaxes[0][2], semiaxes[0][2], 
							args=(semiaxes, r, G_g, rho1, rho2, R1, R2))[0]
	dUdr2 = integrate.dblquad(potential_ellipsoid_derivative_rx, 0, 2*np.pi, -semiaxes2[0][2], semiaxes2[0][2], 
							args=(semiaxes2, -r, G_g, rho1, rho2, R1, R2))[0]
	dUdr = integrate.dblquad(potential_ellipsoid_derivative_spherical, 0, 2*np.pi, -semiaxes[0][2], semiaxes[0][2], 
							args=(semiaxes, r, G_g, rho1, rho2, R1, R2))[0]
	dUdr2 = integrate.dblquad(potential_ellipsoid_derivative_spherical, 0, 2*np.pi, -semiaxes2[0][2], semiaxes2[0][2], 
							args=(semiaxes2, -r, G_g, rho1, rho2, R1, R2))[0]
	"""
	dUdr = integrate.dblquad(potential_ellipsoid_derivative_cylindrical, 0, 2*np.pi, -semiaxes[0][2], semiaxes[0][2], 
							args=(semiaxes, r, G_g, rho1, rho2, R1, R2))[0]
	dUdr2 = integrate.dblquad(potential_ellipsoid_derivative_cylindrical, 0, 2*np.pi, -semiaxes2[0][2], semiaxes2[0][2], 
							args=(semiaxes2, -r, G_g, rho1, rho2, R1, R2))[0]
	print(dUdr)
	print(dUdr2)
	#print("Diff in integral = ", dUdr2-dUdr)
	omega_0_sq = (m1+m2)*np.abs(dUdr)/(m1*m2*np.linalg.norm(r))
	omega_0_sq2 = (m1+m2)*np.abs(dUdr2)/(m1*m2*np.linalg.norm(r))
	#print(np.sqrt(omega_0_sq))
	#print(np.sqrt(omega_0_sq2))
	
	"""
	"""
	#sys.exit(0)
	#print((m1+m2)/(m1*m2*np.linalg.norm(r)))
	#print(dUdr)
	return np.sqrt(omega_0_sq), np.sqrt(omega_0_sq2)

def get_minimum_ellipse_distance(init_xo, main_parameters, theta_0):
	"""
	Moves rotated ellipses such that they are separated by a distance dmin.
	Only rotation for one of the ellipses.
	First ellipse assumed to be non-rotated.
	Rotation only about one axis.

	Recommended to rescale variables so decimals are not used.
	E.g. distance of 1e-5 km should be rescaled to 1km, and every other variables,
	such as semiaxes, should also be rescaled.
	"""
	def min_func(x, d):
		# x[0] = x0
		# x[1], x[2] = x1, y1
		# x[3], x[4] = x2, y2
		return (x[1]-x[3])**2 + (x[2]-x[4])**2 - d**2
		
	def constraint1(x, a, b):
		return (x[1]/a)**2 + (x[2]/b)**2 - 1

	def constraint2(x, a, b, theta):
		xprime = (x[3]-x[0])*np.cos(theta) + x[4]*np.sin(theta)
		yprime = -(x[3]-x[0])*np.sin(theta) + x[4]*np.cos(theta)
		
		term1 = (xprime/a)**2
		term2 = (yprime/b)**2
		return term1 + term2 - 1

	def constraint3(x):
		return x[3]

	def constraint4(x, a_p):
		return x[0] - a_p

	def constraint5(x, a_p, b_p):
		return (x[4]-x[2])*(-x[1])*b_p**2 + (x[3]-x[1])*x[2]*a_p**2

	def constraint6(x, a_s, b_s, theta):
		xprime = (x[3]-x[0])*np.cos(theta) + x[4]*np.sin(theta)
		yprime = -(x[3]-x[0])*np.sin(theta) + x[4]*np.cos(theta)
		
		term1 = (-xprime*b_s**2*np.cos(theta) + yprime*a_s**2*np.sin(theta))*(x[4]-x[2])
		term2 = (xprime*b_s**2*np.sin(theta) + yprime*a_s**2*np.cos(theta))*(x[3]-x[1])
		return term1 + term2

	dmin = main_parameters[0]
	a_p = main_parameters[1]
	b_p = main_parameters[2]
	a_s = main_parameters[3]
	b_s = main_parameters[4]

	cons = ({'type': 'eq', 'fun': constraint1, 'args': (a_p, b_p)},
			{'type': 'eq', 'fun': constraint2, 'args': (a_s, b_s, -theta_0)},
			{'type': 'ineq', 'fun': constraint3},
			{'type': 'ineq', 'fun': constraint4, 'args': (a_p,)},
			{'type': 'eq', 'fun': min_func, 'args': (dmin,)},
			{'type': 'eq', 'fun': constraint5, 'args': (a_p, b_p)},
			{'type': 'eq', 'fun': constraint6, 'args': (a_s, b_s, -theta_0)},
			)

	res = minimize(min_func, init_xo, args=(dmin,), method='SLSQP', constraints=cons)
	return res

def normal_vector_triangle_RH(v0, v1, v2, v3):
	r1 = v1 - v0
	r2 = v2 - v0
	r3 = v3 - v0
	cprod = np.cross(r2, r1)
	n_temp = cprod/np.linalg.norm(cprod)
	d_check = np.dot(r3, n_temp)
	if d_check > 0:
		return -n_temp
	else:
		return n_temp

def polyhedron_grav_field(r, v1, v2, v3, n_i, rhoG):
	def get_I_comp(r, r_ij, r_ij_2, n_i):
		common_denom = np.linalg.norm(r_ij_2 - r_ij)
		a_ij = np.linalg.norm(r - r_ij)/common_denom
		b_ij = np.dot(r - r_ij, r_ij_2 - r_ij)/common_denom**2
		c_ij = np.dot(n_i, r - r_ij)/common_denom
		d_ij = np.dot(np.cross(n_i, r - r_ij), r_ij_2 - r_ij)/common_denom
		K1 = np.sqrt(a_ij**2 - b_ij**2 - c_ij**2)
		K2 = np.sqrt(1 + a_ij**2 - 2*b_ij)
		I = d_ij*((c_ij/K1)*(np.arctan(c_ij*(1 - b_ij)/(K1*K2)) + np.arctan(c_ij*b_ij/(a_ij*K1))) 
				  + np.log((1 - b_ij + K2)/(a_ij-b_ij)))
		return I

	def get_K_comp(r, r_ij, r_ij_2, n_i):
		signfunc = 0
		r_p = np.dot(n_i, r_ij)*(n_i - np.cross(np.cross(n_i, r), n_i))
		frontfac = np.dot(n_i, np.cross(r_ij - r_p, r_ij_2 - r_p))
		backfac = np.arccos(np.dot(r_ij - r_p, r_ij_2 - r_p)/(np.linalg.norm(r_ij - r_p)*np.linalg.norm(r_ij_2 - r_p)))
		if frontfac < 0:
			signfunc = -1
		elif frontfac > 0:
			signfunc = 1
		else:
			signfunc = 1
		theta_ij = frontfac*backfac
		K = -np.linalg.norm(np.dot(n_i, r - r_ij))*theta_ij
		return K

	I1 = get_I_comp(r, v1, v2, n_i)
	I2 = get_I_comp(r, v2, v3, n_i)
	I3 = get_I_comp(r, v3, v1, n_i)
	Isum = I1 + I2 + I3
	K1 = get_K_comp(r, v1, v2, n_i)
	K2 = get_K_comp(r, v2, v3, n_i)
	K3 = get_K_comp(r, v3, v1, n_i)
	Ksum = K1 + K2 + K3

	g_x = -rhoG*n_i[0]*(Isum + Ksum)
	g_y = -rhoG*n_i[1]*(Isum + Ksum)
	g_z = -rhoG*n_i[2]*(Isum + Ksum)
	return g_x, g_y, g_z

def ellipsoid_touch_secondary_cord(a_p, b_p, c_p, a_s, b_s, c_s, x_i, y_i, z_i):
	# Algorithm that determines the coordinate of the centroid of a secondary ellipsoid
	# so that the surfaces of the two ellipsoids touch.
	# Primary ellipsoid assumed to be at the origin.
	# x_i, y_i, z_i are surface points on the primary where the bodies touch
	def equationsystem(variables, params):
		x_0, y_0, z_0, lamd = variables
		a_p, b_p, c_p, a_s, b_s, c_s, x_i, y_i, z_i = params
		eq1 = (x_i - x_0)*(a_p/a_s)**2 - lamd*x_i
		eq2 = (y_i - y_0)*(b_p/b_s)**2 - lamd*y_i
		eq3 = (z_i - z_0)*(c_p/c_s)**2 - lamd*z_i
		eq4 = x_0*x_i/a_s**2 + y_0*y_i/b_s**2 + z_0*z_i/c_s**2 - ((x_0/a_s)**2 + (y_0/b_s)**2 + (z_0/c_s)**2 - 1) - lamd
		return [eq1, eq2, eq3, eq4]
	in_params = [a_p, b_p, c_p, a_s, b_s, c_s, x_i, y_i, z_i]
	#sol = fsolve(equationsystem, (x_i+a_s, y_i+b_s, z_i+c_s, 0), args=(in_params))
	x_0, y_0, z_0, lambd = fsolve(equationsystem, (x_i+a_s, y_i+b_s, z_i+c_s, 0), args=(in_params))
	x_02 = x_i*(1 + lambd*(a_s/a_p)**2)
	y_02 = y_i*(1 + lambd*(b_s/b_p)**2)
	z_02 = z_i*(1 + lambd*(c_s/c_p)**2)
	if np.sqrt(x_0**2 + y_0**2 + z_0**2) > np.sqrt(x_02**2 + y_02**2 + z_02**2):
		return x_0, y_0, z_0
	else:
		return x_02, y_02, z_02

def total_angular_momentum_2bod(sol, semiaxes, massdens, Use_euler_params):
	a_A = semiaxes[0][0]
	b_A = semiaxes[0][1]
	c_A = semiaxes[0][2]
	a_B = semiaxes[1][0]
	b_B = semiaxes[1][1]
	c_B = semiaxes[1][2]
	ma = massdens[0]
	mb = massdens[1]
	rho_A = massdens[2]
	rho_B = massdens[3]

	I_A = OF.moment_inertia_ellipsoid(a_A, b_A, c_A, rho_A)
	I_B = OF.moment_inertia_ellipsoid(a_B, b_B, c_B, rho_B)
	I_11_A = I_A[0][0]
	I_22_A = I_A[1][1]
	I_33_A = I_A[2][2]
	I_11_B = I_B[0][0]
	I_22_B = I_B[1][1]
	I_33_B = I_B[2][2]
	
	NNN = len(sol.t)
	wxA = np.zeros(NNN)
	wyA = np.zeros(NNN)
	wzA = np.zeros(NNN)
	wxB = np.zeros(NNN)
	wyB = np.zeros(NNN)
	wzB = np.zeros(NNN)
	for i in range(NNN):
		if Use_euler_params:
			R_A = OF.rotation_matrix_euler(sol.y[18][i],sol.y[19][i],sol.y[20][i], sol.y[21][i])
			R_B = OF.rotation_matrix_euler(sol.y[22][i],sol.y[23][i],sol.y[24][i], sol.y[25][i])
		else:
			R_A = OFcy.rotation_matrix(sol.y[18][i],sol.y[19][i],sol.y[20][i])
			R_B = OFcy.rotation_matrix(sol.y[21][i],sol.y[22][i],sol.y[23][i])
		IomegaA = np.array([I_11_A*sol.y[12][i],I_22_A*sol.y[13][i],I_33_A*sol.y[14][i]])
		IomegaB = np.array([I_11_B*sol.y[15][i],I_22_B*sol.y[16][i],I_33_B*sol.y[17][i]])
		
		wxA[i],wyA[i],wzA[i] = np.dot(R_A,IomegaA)
		wxB[i],wyB[i],wzB[i] = np.dot(R_B,IomegaB)	
	
	Rcm_x = (ma*sol.y[6] + mb*sol.y[9])/(ma+mb)
	Rcm_y = (ma*sol.y[7] + mb*sol.y[10])/(ma+mb)
	Rcm_z = (ma*sol.y[8] + mb*sol.y[11])/(ma+mb)
	# Velocity of centre of mass. Should be zero/constant
	Vcm_x = (ma*sol.y[0] + mb*sol.y[3])/(ma+mb)
	Vcm_y = (ma*sol.y[1] + mb*sol.y[4])/(ma+mb)
	Vcm_z = (ma*sol.y[2] + mb*sol.y[5])/(ma+mb)
	# Positions of A and B in the CM reference frame
	xA = sol.y[6] - Rcm_x
	yA = sol.y[7] - Rcm_y
	zA = sol.y[8] - Rcm_z
	xB = sol.y[9] - Rcm_x
	yB = sol.y[10] - Rcm_y
	zB = sol.y[11] - Rcm_z
	#xA, yA, zA = np.dot(R_A, np.array([xA, yA, zA]))
	#xB, yB, zB = np.dot(R_B, np.array([xB, yB, zB]))
	vxA = sol.y[0] - Vcm_x
	vyA = sol.y[1] - Vcm_y
	vzA = sol.y[2] - Vcm_z
	vxB = sol.y[3] - Vcm_x
	vyB = sol.y[4] - Vcm_y
	vzB = sol.y[5] - Vcm_z
	#vxA, vyA, vzA = np.dot(R_A, np.array([vxA, vyA, vzA]))
	#vxB, vyB, vzB = np.dot(R_B, np.array([vxB, vyB, vzB]))

	h_Ax = ma*(yA*vzA - zA*vyA) + wxA
	h_Ay = ma*(zA*vxA - xA*vzA) + wyA
	h_Az = ma*(xA*vyA - yA*vxA) + wzA
	
	h_Bx = mb*(yB*vzB - zB*vyB) + wxB
	h_By = mb*(zB*vxB - xB*vzB) + wyB
	h_Bz = mb*(xB*vyB - yB*vxB) + wzB

	Jtot_x = h_Ax + h_Bx
	Jtot_y = h_Ay + h_By
	Jtot_z = h_Az + h_Bz
	return Jtot_x, Jtot_y, Jtot_z

def center_of_mass_pos(sol, masses, N_bodies):
	""" Center of mass positions between two bodies """
	Rcm_x = 0
	Rcm_y = 0
	Rcm_z = 0
	for i in range(N_bodies):
		Rcm_x += (masses[i]*sol.y[N_bodies*3 + i*3])
		Rcm_y += (masses[i]*sol.y[N_bodies*3 + i*3 + 1])
		Rcm_z += (masses[i]*sol.y[N_bodies*3 + i*3 + 2])
	Rcm_x /= np.sum(masses)
	Rcm_y /= np.sum(masses)
	Rcm_z /= np.sum(masses)
	xcm = np.zeros((N_bodies, len(sol.t)))
	ycm = np.zeros((N_bodies, len(sol.t)))
	zcm = np.zeros((N_bodies, len(sol.t)))
	for i in range(N_bodies):
		xcm[i] = sol.y[N_bodies*3 + i*3] - Rcm_x
		ycm[i] = sol.y[N_bodies*3 + i*3 + 1] - Rcm_y
		zcm[i] = sol.y[N_bodies*3 + i*3 + 2] - Rcm_z

	return xcm, ycm, zcm

if __name__ == '__main__':
	# Testing functions
	value = kappa_derivative(1,1,3.0,0,0,9,0,0,1)
	value2 = kappa_derivative(1,1,3.0,0,0,9,0,0,1,positive_root=False)
	print(value)
	print(value2)

	g_value = g_field(1.0,1.0,-3.3,0,0,value,100,2.0)
	print(g_value)
	