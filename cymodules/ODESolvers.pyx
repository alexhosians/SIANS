"""
The ODE class solver.
This script contains various of numerical integrators.
"""

# cython: linetrace=True
# Basic imports
import sys
import time
# Numpy and scipy.integrate import
import numpy as np
cimport numpy as np
## Cython functions
import cython
# Import C functions
from libc.math cimport pi,sqrt,cos,sin,fabs
from libc.stdlib cimport malloc, free
# Import own modules
cimport odetableaus

class ODESolvers:
	""" 
	This class is used to call the Cython class ODESolvers_main.
	The main purpose of this class is to wrap the results into something pure-python can call.
	Use as:
	>> import Cython.ODESolvers as ODESolvers
	>> sol = ODESolvers.ODESolvers(args*)

	sol will then be an instance with the global ellipsoid parameters as attributes.
	For instance, to get the x-positions, call
	>> x_pos = sol.X
	If you want the positions of ellipsoid 0 and 1, call
	>> x_pos_0 = sol.X[0]
	>> x_pos_1 = sol.X[1]
	>> x_pos_n = sol.X[n]

	If you want to convert the results similar to scipy's solve_ivp solution, then
	>> sol = ODESolvers.ODESolvers(tspan, y_init, semiaxes, masses, NBodies, method)
	>> sol.sort_as_ysol()
	>> x_pos_0 = sol.y[NBodies*3 + i*0]
	>> x_pos_1 = sol.y[NBodies*3 + i*1]
	>> x_pos_n = sol.y[NBodies*3 + i*n]
	"""
	
	def __init__(self, double[:] tspan, double[:] y_init, double[:,:] semiaxes, double[:] vertices, int[:] p_info, int[:] face_ids,
		double[:] masses, int[:] int_info, double[:] tolerance_arr, double G_grav, str SaveFolder, method='RK54', int savemid=0):
		"""
		Calls ODE solver from python.
		When results are computed, sorts then properly.
		"""
		result = ODESolver(tspan, y_init, semiaxes, vertices, p_info, face_ids, masses, int_info, tolerance_arr, G_grav,
									 SaveFolder, method=method)
		self.t = result.tspan.base
		self.nfev = result.nfev
		self.NBodies = int_info[0]
		self.NSolFiles = result.NSolFiles
		self.EulerParams = int_info[1]
		self.crash = result.crashcheck
		if self.NSolFiles == 1:
			self.vx = result._vx.base.transpose()
			self.vy = result._vy.base.transpose()
			self.vz = result._vz.base.transpose()
			self.x = result._x.base.transpose()
			self.y = result._y.base.transpose()
			self.z = result._z.base.transpose()
			self.p = result._p.base.transpose()
			self.q = result._q.base.transpose()
			self.r = result._r.base.transpose()
			if self.EulerParams:
				self.e0 = result._e0.base.transpose()
				self.e1 = result._e1.base.transpose()
				self.e2 = result._e2.base.transpose()
				self.e3 = result._e3.base.transpose()
			else:
				self.phi = result._phi.base.transpose()
				self.theta = result._theta.base.transpose()
				self.psi = result._psi.base.transpose()
		else:
			print("More than one solution file. Does not return a solution instance with solutions.")

	def sort_as_ysol(self):
		""" Sorts solution files as similar to the sol instance from scipy.solve_ivp"""
		cdef int i, C_i, D_i
		#if self.nfev < self.vx.shape[0]:
		#	numsteps = self.nfev
		cdef int pos_idx = self.NBodies*3
		cdef int angvel_idx = self.NBodies*6
		cdef int ang_idx = self.NBodies*9
		if self.NSolFiles == 1:
			if self.EulerParams:
				self.var = np.zeros((13*self.NBodies,self.nfev), dtype=np.float64)
			else:
				self.var = np.zeros((12*self.NBodies,self.nfev), dtype=np.float64)
			self.t = self.t[:self.nfev]
			for i in range(self.NBodies):
				C_i = 3*i
				D_i = 4*i
				self.var[C_i] = self.vx[i][:self.nfev]
				self.var[C_i + 1] = self.vy[i][:self.nfev]
				self.var[C_i + 2] = self.vz[i][:self.nfev]
				self.var[pos_idx + C_i] = self.x[i][:self.nfev]
				self.var[pos_idx + C_i + 1] = self.y[i][:self.nfev]
				self.var[pos_idx + C_i + 2] = self.z[i][:self.nfev]
				self.var[angvel_idx + C_i] = self.p[i][:self.nfev]
				self.var[angvel_idx + C_i + 1] = self.q[i][:self.nfev]
				self.var[angvel_idx + C_i + 2] = self.r[i][:self.nfev]
				if self.EulerParams:
					self.var[ang_idx + D_i] = self.e0[i][:self.nfev]
					self.var[ang_idx + D_i + 1] = self.e1[i][:self.nfev]
					self.var[ang_idx + D_i + 2] = self.e2[i][:self.nfev]
					self.var[ang_idx + D_i + 3] = self.e3[i][:self.nfev]
				else:
					self.var[ang_idx + C_i] = self.phi[i][:self.nfev]
					self.var[ang_idx + C_i + 1] = self.theta[i][:self.nfev]
					self.var[ang_idx + C_i + 2] = self.psi[i][:self.nfev]


cdef class ODESolver:
	"""
	The ODESolvers class. This class contains various of ODE solvers. 
	Following numerical integrators are implemented:

	== Runge-Kutta methods:
	method='RK4' -- Classical Runge Kutta 4th order method.
	method='RK54' -- Dormand-Prince of order 5(4) method (DEFAULT).
	method='V65' -- Runge-Kutta method of order 6(5) due to Verner.
	method='V76' -- Runge-Kutta method of order 7(6) due to Verner.
	method='V87' -- Runge-Kutta method of order 8(7) due to Verner.
	method='V98' -- Runge-Kutta method of order 9(8) due to Verner.
	method='T98' -- Runge-Kutta method of order 9(8) due to Tsitouras.
	method='F108' -- Runge-Kutta method of order  10(8) due to T. Feagin.
	method='F1210' -- Runge-Kutta method of order  12(10) due to T. Feagin.
	"""
	# Define class-wide quantities
	cdef double[:] tspan
	cdef double[:] semiaxes
	cdef double[:] masses
	cdef double[:] y_init
	cdef double[:] vertices
	cdef double[:,:] _vx, _vy, _vz
	cdef double[:,:] _x, _y, _z
	cdef double[:,:] _p, _q, _r
	cdef double[:,:] _phi, _theta, _psi
	cdef double[:,:] _e0, _e1, _e2, _e3
	cdef int[:] p_info, face_ids
	cdef int N_steps, N_steps_OG, NumBodies, nfev, MaxPts, NSolFiles, EulerParams, includeSun, crashcheck, docrashcheck
	cdef double ntol, iabstol, ireltol, quadval, quadnode, hmax, hmin, G_grav
	cdef double t0, tf
	def __init__(self, double[:] tspan, double[:] y_init, double[:,:] semiaxes, double[:] vertices, int[:] p_info, int[:] face_ids,
	 	double[:] masses, int[:] int_info, double[:] tolerance_arr, double G_grav, str SaveFolder, str method='RK54', int savemid=0):
		""" Initializes the solver. Runs solver and returns solutions."""
		# Some checks before computation
		cdef int NBodies = int_info[0]
		cdef int EulerParams = int_info[1]
		cdef int includeSun = int_info[2]
		cdef int useAdaptive = int_info[3]
		if NBodies <= 1:
			sys.exit("NBodies must be greater than one.")
		if tspan.shape[0] <= 1:
			sys.exit("Timespan must have more than 1 evaluation point.")
		if y_init.shape[0] <= 0:
			sys.exit("y_init array is empty.")
		if EulerParams:
			if not (y_init.shape[0] == NBodies*13):
				sys.exit("y_init shape not consistent with total number of variables.")
		else:
			if not (y_init.shape[0] == NBodies*12):
				sys.exit("y_init shape not consistent with total number of variables.")
		if p_info.shape[0] != 2*NBodies:
			sys.exit("Shape of p_info not equal to number of bodies")

		self.NumBodies = NBodies
		N_steps = tspan.shape[0]
		self.t0 = tspan[0]
		self.tf = tspan[-1]
		self.nfev = N_steps
		self.ntol = tolerance_arr[0]
		self.iabstol = tolerance_arr[1]
		self.ireltol = tolerance_arr[2]
		self.quadval = tolerance_arr[3]
		self.hmax = tolerance_arr[4]
		self.hmin = tolerance_arr[5]
		self.includeSun = includeSun
		self.y_init = y_init
		self.semiaxes = np.zeros(3*NBodies, dtype=np.float64)
		cdef int i
		for i in range(NBodies):
			self.semiaxes[3*i] = semiaxes[i][0]
			self.semiaxes[3*i+1] = semiaxes[i][1]
			self.semiaxes[3*i+2] = semiaxes[i][2]
			
		self.p_info = p_info
		self.face_ids = face_ids
		self.vertices = vertices
		self.masses = masses
		self.G_grav = G_grav
		self.NSolFiles = 1
		# Set maximum number of data points based on number of bodies
		# This is to perserve memory usage
		# Can be manually edited here, but files must be recompiled.
		if NBodies <= 4:
			self.MaxPts = 500000
		elif NBodies > 4 and NBodies <= 15:
			self.MaxPts = 100000
		elif NBodies > 15: # Current max. If more bodies are added, add more checks
			self.MaxPts = 50000
		else:
			sys.exit("Number of bodies = %d. Must be positive and larger than 0!" %NBodies)
		self.N_steps_OG = N_steps
		if N_steps > self.MaxPts:
			N_steps = self.MaxPts
		self.N_steps = N_steps
		self.EulerParams = EulerParams

		# Initialize empty arrays that saves the variables
		self._vx = np.zeros((N_steps,NBodies))     # X-velocities
		self._vy = np.zeros((N_steps,NBodies))     # Y-velocities
		self._vz = np.zeros((N_steps,NBodies))     # Z-velocities
		self._x = np.zeros((N_steps,NBodies))      # X-positions
		self._y = np.zeros((N_steps,NBodies))      # Y-positions
		self._z = np.zeros((N_steps,NBodies))      # Z-positions
		self._p = np.zeros((N_steps,NBodies))      # X-angular speed
		self._q = np.zeros((N_steps,NBodies))      # Y-angular speed
		self._r = np.zeros((N_steps,NBodies))      # Z-angular speed
		if EulerParams:
			self._e0 = np.zeros((N_steps, NBodies))
			self._e1 = np.zeros((N_steps, NBodies))
			self._e2 = np.zeros((N_steps, NBodies))
			self._e3 = np.zeros((N_steps, NBodies))
		else:
			self._phi = np.zeros((N_steps,NBodies))    # Rotation around x
			self._theta = np.zeros((N_steps,NBodies))  # Rotation around y
			self._psi = np.zeros((N_steps,NBodies))    # Rotation around z
		if useAdaptive:
			self.tspan = np.zeros(N_steps)
		else:
			self.tspan = np.linspace(self.t0, self.tf, N_steps)
		# Sets initial values
		self.set_initial_values(y_init, NBodies, EulerParams)
		self.RK_adaptive(SaveFolder, method=method, useAdaptive=useAdaptive, savemid=savemid)

	cpdef double set_initial_values(self, double[:] y_init, int NBodies, int EulerParams):
		""" 
		A function that sets the initial values to the respective arrays.

		Parameters:
		--- y_init:   Initial values of the system for all bodies.
		--- N_bodies: Number of bodies in the system.
		"""
		cdef Py_ssize_t i
		cdef int C_i   		  # Used to pick out the cordinate component for i'th body
		# Variables used as a 'jump' to access variable indices of y_init
		cdef int pos_idx = NBodies*3
		cdef int angvel_idx = NBodies*6
		cdef int ang_idx = NBodies*9
		with cython.boundscheck(False):
			if EulerParams:
				for i in range(NBodies):
					C_i = i*3
					self._vx[0][i] = y_init[C_i + 0]
					self._vy[0][i] = y_init[C_i + 1]
					self._vz[0][i] = y_init[C_i + 2]
					self._x[0][i] = y_init[pos_idx + C_i + 0]
					self._y[0][i] = y_init[pos_idx + C_i + 1]
					self._z[0][i] = y_init[pos_idx + C_i + 2]
					self._p[0][i] = y_init[angvel_idx + C_i + 0]
					self._q[0][i] = y_init[angvel_idx + C_i + 1]
					self._r[0][i] = y_init[angvel_idx + C_i + 2]
					self._e0[0][i] = y_init[ang_idx + i*4]
					self._e1[0][i] = y_init[ang_idx + i*4 + 1]
					self._e2[0][i] = y_init[ang_idx + i*4 + 2]
					self._e3[0][i] = y_init[ang_idx + i*4 + 3]
			else:
				for i in range(NBodies):
					C_i = i*3
					self._vx[0][i] = y_init[C_i + 0]
					self._vy[0][i] = y_init[C_i + 1]
					self._vz[0][i] = y_init[C_i + 2]
					self._x[0][i] = y_init[pos_idx + C_i + 0]
					self._y[0][i] = y_init[pos_idx + C_i + 1]
					self._z[0][i] = y_init[pos_idx + C_i + 2]
					self._p[0][i] = y_init[angvel_idx + C_i + 0]
					self._q[0][i] = y_init[angvel_idx + C_i + 1]
					self._r[0][i] = y_init[angvel_idx + C_i + 2]
					self._phi[0][i] = y_init[ang_idx + C_i + 0]
					self._theta[0][i] = y_init[ang_idx + C_i + 1]
					self._psi[0][i] = y_init[ang_idx + C_i + 2]

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef int update_arrays(self, double[:] y, double current_time, int step, str SaveFolder):
		""" 
		Updates the compute variables as well as time.
		For adaptive time steppers, will expand the size of the parameter arrays
		if number evaluations exceeds number of array elements in the respective parameters.
		
		If the step exceeds maximum number of array points, saves the computed variables
		to a numpy file. The step counter is reset to 0.
		"""
		cdef Py_ssize_t i
		cdef int C_i, D_i   		  # Used to pick out the cordinate component for i'th body
		# Variables used as a 'jump' to access variable indices of y_init
		cdef int NBodies = self.NumBodies
		cdef int pos_idx = NBodies*3
		cdef int angvel_idx = NBodies*6
		cdef int ang_idx = NBodies*9
		if step >= self.N_steps and step < self.MaxPts:
			self.realloc_solution_arrs(NBodies)
		
		if self.EulerParams:
			for i in range(NBodies):
				C_i = i*3
				D_i = i*4
				self._vx[step][i] = y[C_i]
				self._vy[step][i] = y[C_i + 1]
				self._vz[step][i] = y[C_i + 2]
				self._x[step][i] = y[pos_idx + C_i]
				self._y[step][i] = y[pos_idx + C_i + 1]
				self._z[step][i] = y[pos_idx + C_i + 2]
				self._p[step][i] = y[angvel_idx + C_i]
				self._q[step][i] = y[angvel_idx + C_i + 1]
				self._r[step][i] = y[angvel_idx + C_i + 2]
				self._e0[step][i] = y[ang_idx + D_i]
				self._e1[step][i] = y[ang_idx + D_i + 1]
				self._e2[step][i] = y[ang_idx + D_i + 2]
				self._e3[step][i] = y[ang_idx + D_i + 3]
		else:
			for i in range(NBodies):
				C_i = i*3
				self._vx[step][i] = y[C_i]
				self._vy[step][i] = y[C_i + 1]
				self._vz[step][i] = y[C_i + 2]
				self._x[step][i] = y[pos_idx + C_i]
				self._y[step][i] = y[pos_idx + C_i + 1]
				self._z[step][i] = y[pos_idx + C_i + 2]
				self._p[step][i] = y[angvel_idx + C_i]
				self._q[step][i] = y[angvel_idx + C_i + 1]
				self._r[step][i] = y[angvel_idx + C_i + 2]
				self._phi[step][i] = y[ang_idx + C_i]
				self._theta[step][i] = y[ang_idx + C_i + 1]
				self._psi[step][i] = y[ang_idx + C_i + 2]
		self.tspan[step] = current_time
		
		if step == self.MaxPts - 1:
			# Save to solution file here
			self.save_multi_solution(SaveFolder, step)
			step = 0
			self.NSolFiles += 1
		else:
			step += 1
		return step

	cdef int realloc_solution_arrs(self, int NBodies):
		""" 
		Reallocates the solution arrays to bigger arrays if it becomes full. 
		Doubles the size everytime this is called
		"""
		cdef double[:,:] new_zero_arr
		if 2*self.N_steps > self.MaxPts:
			new_zero_arr = np.empty((self.MaxPts - self.N_steps, NBodies), dtype=np.float64)	
			self.tspan = np.hstack([self.tspan, np.empty(self.MaxPts - self.N_steps, dtype=np.float64)])
		else:
			new_zero_arr = np.empty((self.N_steps, NBodies), dtype=np.float64)
			self.tspan = np.hstack([self.tspan, np.empty(self.N_steps, dtype=np.float64)])
			self.N_steps *= 2
		self._vx = np.vstack([self._vx, new_zero_arr])
		self._vy = np.vstack([self._vy, new_zero_arr])
		self._vz = np.vstack([self._vz, new_zero_arr])
		self._x = np.vstack([self._x, new_zero_arr])
		self._y = np.vstack([self._y, new_zero_arr])
		self._z = np.vstack([self._z, new_zero_arr])
		self._p = np.vstack([self._p, new_zero_arr])
		self._q = np.vstack([self._q, new_zero_arr])
		self._r = np.vstack([self._r, new_zero_arr])
		if self.EulerParams:
			self._e0 = np.vstack([self._e0, new_zero_arr])
			self._e1 = np.vstack([self._e1, new_zero_arr])
			self._e2 = np.vstack([self._e2, new_zero_arr])
			self._e3 = np.vstack([self._e3, new_zero_arr])
		else:
			self._phi = np.vstack([self._phi, new_zero_arr])
			self._theta = np.vstack([self._theta, new_zero_arr])
			self._psi = np.vstack([self._psi, new_zero_arr])
		return 0
			
	cdef int save_multi_solution(self, str SaveFolder, int step):
		""" 
		Saves computed variables and time to a numpy file 
		when the number of points exceeds maximum array size.
		"""
		cdef int stepp1 = step + 1
		if self.EulerParams:
			asave = np.array([self._vx[:stepp1], self._vy[:stepp1], self._vz[:stepp1],
							  self._x[:stepp1], self._y[:stepp1], self._z[:stepp1],
							  self._p[:stepp1], self._q[:stepp1], self._r[:stepp1], 
							  self._e0[:stepp1], self._e1[:stepp1], self._e2[:stepp1], self._e3[:stepp1]])
		else:
			asave = np.array([self._vx[:stepp1], self._vy[:stepp1], self._vz[:stepp1],
							  self._x[:stepp1], self._y[:stepp1], self._z[:stepp1],
							  self._p[:stepp1], self._q[:stepp1], self._r[:stepp1], 
							  self._phi[:stepp1], self._theta[:stepp1], self._psi[:stepp1]])
		np.savez_compressed(SaveFolder+"Solutions%d" %(self.NSolFiles), self.tspan[:stepp1], asave)
		return 0

	cdef int bodies_crash_fix(self, int step):
		"""
		Discards time steps after a collision has occured.
		"""
		self._vx = self._vx[:step]
		self._vy = self._vy[:step]
		self._vz = self._vz[:step]
		self._x = self._x[:step]
		self._y = self._y[:step]
		self._z = self._z[:step]
		self._p = self._p[:step]
		self._q = self._q[:step]
		self._r = self._r[:step]
		if self.EulerParams:
			self._e0 = self._e0[:step]
			self._e1 = self._e1[:step]
			self._e2 = self._e2[:step]
			self._e3 = self._e3[:step]
		else:
			self._phi = self._phi[:step]
			self._theta = self._theta[:step]
			self._psi = self._psi[:step]
		return 0


	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef int update_t(self, double current_time, int step):
		"""  Updates time array. Mainly used for adaptive time steppers.  """
		self.tspan[step] = current_time

	######
	# SET THESE TO FALSE (True for cdivision) WHEN CODE IS DONE!!!
	######
	@cython.cdivision(False)
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double[:] K_step_RK_pointer(self, double[:] y, double[:,:] K_old, double[:] a, double c,
		Py_ssize_t stage, int numvars, double current_t, double dt, double* dy_ptr, double* dy_dt_ptr, double* main_prop_ptr,
		double* semiaxes_ptr, double itols[3], double* vertices, int *face_ids, double* moment_inertia, int triggers[3]):
		""" 
		Function that computes the K array for one stage.
		This function uses pointers to compute the differential equations.
		The function that contains the differential equations must be a C function.
		"""
		cdef Py_ssize_t k, i
		cdef double result
		cdef double[:] K_new = np.zeros(numvars, dtype=np.float64)
		for k in range(numvars):
			result = 0.0
			for i in range(stage+1):
				result += K_old[i,k]*a[i]*dt
			dy_ptr[k] = result + y[k]
		self.crashcheck = ode_solver_Nbody(current_t + c*dt, dy_ptr, dy_dt_ptr, main_prop_ptr, semiaxes_ptr, itols, vertices, 
							  face_ids, moment_inertia, triggers)
		if self.crashcheck == -1:
			self.docrashcheck = 1
		elif self.crashcheck == -2:
			self.docrashcheck = -1
			
		for k in range(numvars):
			K_new[k] = dy_dt_ptr[k]
		return K_new

	cdef int RK_adaptive(self, str SaveFolder, str method='RK54', bint useAdaptive=True, int savemid=0):
		""" 
		A Runge-Kutta method with an adaptive time stepper.
		Parameters:
		--- method: (string)
			Parameter used to determine the solver used.
			Defaults to Dormand-Prince of order 5(4).

		Numerical methods currently implemented:
		Runge Kutta 4  -- Classical Runge Kutta order 4 method.
		Dormand Prince 5(4)  -- Runge-Kutta method of order 5(4) due to Dormand and Prince. 
					  		    This is equivalent to scipy's RK54 method, but this is roughly
					  		    2 orders of magnitude more accurate.
					  		    This is the default method.
		Verner 6(5) -- Runge-Kutta method of order 6(5) due to Verner.
		Verner 7(6) -- Runge-Kutta method of order 7(6) due to Verner.
					   A 6th order dense output version of this method is also available.
		Dormand Prince 8(6) -- Runge-Kutta method of order 8(6) due to Dormand and Prince.
							   Current implementation does not work properly.
		Verner 8(7) -- Runge-Kutta method of order 8(7) due to Verner.
		Verner 9(8) -- Runge-Kutta method of order 9(8) due to Verner.
					   An 8th order dense output version of this method is also available.
		Tsitouras 9(8) -- Runge-Kutta method of order 9(8) due to Tsitouras.
		Feagin 10(8) -- Runge-Kutta method of order  10(8) due to T. Feagin.
		Feagin 12(10) -- Runge-Kutta method of order  12(10) due to T. Feagin.

		Notes on the Butcher Tableau parameters:
		- The C array contains all the nodes of the Butcher Tableau
		- The A matrix contains the coupling coefficients
		- The B array contains the higher order weights
		- The E array contains the lower order weights
		- p_order is the order of the method, used for adaptive time stepping
		- fac is the safety factor for adaptive time stepping
		"""
		cdef double[:] u = np.zeros(1, dtype=np.float64)
		cdef int n_stages = odetableaus.Getstages(method, u)
		cdef double[:] params = np.zeros(2, dtype=np.float64)  # Stores [p_order, fac]
		# Arrays for the Butcher Tableau coefficients
		cdef double[:,:] A = np.zeros((n_stages-1,n_stages-1), dtype=np.float64)
		cdef double[:] C = np.zeros(n_stages-1, dtype=np.float64)
		cdef double[:] B = np.zeros(n_stages, dtype=np.float64)
		cdef double[:] E = np.zeros(n_stages, dtype=np.float64)
		cdef int errorchecks = odetableaus.SelectSolver(method, A, C, B, E, params, u[0])
		if not useAdaptive and errorchecks < 4:
			errorchecks = 4

		if errorchecks < 4:
			self.Solve_RK_adaptive(n_stages, A, C, B, E, params, errorchecks, SaveFolder, savemid)
			#if self.includeSun == 1:
			#	self.Solve_RK_pointer_wSun(n_stages, A, C, B, E, params, errorchecks, SaveFolder, savemid)
			#else:		
		elif errorchecks == 4:
			self.RK_non_adaptive(n_stages, A, C, B, E, SaveFolder, savemid)
			#self.Rotational_fission_test(n_stages, A, C, B, E, SaveFolder, savemid)
			#if self.includeSun == 1:
			#	self.Dense_Output_wSun(n_stages, A, C, B, E, SaveFolder, savemid)
			#else:
		else:
			raise ValueError("Value of errorchecks not ok!")
		return 0

	######
	# SET THESE TO FALSE (True for cdivision) WHEN CODE IS DONE!!!
	######
	@cython.cdivision(False)
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef int Solve_RK_adaptive(self, int n_stages, double[:,:] A, double[:] C, double[:] B, double[:] E,
		double[:] params, int errorchecks, str SaveFolder, int savemid):
		"""
		Function that steps the Runge-Kutta method.
		Adaptive time stepper and error estimator are implemented in this function.
		
		If errorchecks = 1: Uses error estimator and adaptive time stepper described in Harris et al.
		See: https://www.springer.com/gp/book/9783540566700
		
		If errorchecks = 2: Uses error estimator and adaptive time stepper described in Tsitouras (2011).
		See: https://doi.org/10.1016/j.camwa.2011.06.002.

		This function utilizes pointers to send and receive data from the differential equations.
		The pointers are specifically used for C-functions.
		For a memoryview and Cython usage, see the alternative function.
		"""
		cdef double tolerance = self.ntol
		cdef double hmax = self.hmax
		cdef double hmin = self.hmin
		cdef double p_order = params[0]
		cdef double fac = params[1]
		cdef double p_order_fac = 1.0/p_order
		cdef double[3] itols = [self.iabstol, self.ireltol, self.quadval]

		cdef Py_ssize_t i, C_i
		cdef int counter = 1
		cdef int step_counter = 1
		cdef double h, localErrorNorm, localerr
		cdef int numvars
		cdef int[3] triggers = [self.EulerParams, self.includeSun, self.NumBodies]
		if self.EulerParams:
			numvars = self.NumBodies*13
		else:
			numvars = self.NumBodies*12
		cdef double[:,:] K = np.zeros((n_stages,numvars),dtype=np.float64)
		cdef double rtol = tolerance
		cdef double atol = tolerance   # Both atol and rtol should be small for adaptive time stepper to work!
		cdef double t0 = self.t0
		cdef double tf = self.tf
		cdef double dt = (tf-t0)/(self.N_steps_OG-1)
		cdef double current_t = t0
		cdef double[:] y_tild
		cdef double[:] y_temp
		cdef double[:] y = self.y_init
		# Compute densities
		cdef int N_vertices = 0
		for i in range(self.NumBodies):
			N_vertices += self.p_info[i]*3
		# Pointers to store data used in the C function
		cdef double* y_ptr = <double *>malloc(numvars*sizeof(double))
		cdef double* dy_ptr = <double *>malloc(numvars*sizeof(double))
		cdef double* dy_dt_ptr = <double *>malloc(numvars*sizeof(double))
		cdef double* main_prop_ptr = <double *>malloc((self.NumBodies*5 + 1)*sizeof(double))
		cdef double* semiaxes_ptr = <double *>malloc((self.NumBodies*3)*sizeof(double))
		cdef double* vertices = <double *>malloc(N_vertices*sizeof(double))
		cdef double* moment_inertia = <double *>malloc(self.NumBodies*9*sizeof(double))
		for i in range(numvars):
			y_ptr[i] = y[i]
		for i in range(N_vertices):
			vertices[i] = self.vertices[i]
		for i in range(self.NumBodies):
			semiaxes_ptr[3*i] = self.semiaxes[3*i]
			semiaxes_ptr[3*i+1] = self.semiaxes[3*i+1]
			semiaxes_ptr[3*i+2] = self.semiaxes[3*i+2]
			main_prop_ptr[self.NumBodies*4 + i] = self.p_info[i]
		main_prop_ptr[self.NumBodies*5] = self.G_grav
		cdef int total_faces = 0
		for i in range(self.NumBodies):
			total_faces += self.p_info[self.NumBodies + i]
		cdef int* face_ids_all = <int *>malloc(3*total_faces*sizeof(int))
		for i in range(3*total_faces):
			face_ids_all[i] = self.face_ids[i]

		Sort_mainprop_pointer_polyhedron(main_prop_ptr, vertices, self.NumBodies, self.p_info, self.face_ids, 
										 self.masses, self.semiaxes, moment_inertia)
		cdef int adaptive_hit = 0
		cdef int tolcheck
		cdef double err_factor = errfac_Feagin(n_stages)
		cdef double time_counter = 1.0
		cdef double progress_checker = 0.1*time_counter*tf
		cdef double time_start = time.time()
		cdef int reduce_step
		self.crashcheck = 0
		self.docrashcheck = 0
		if not (errorchecks == 1 or errorchecks == 2 or errorchecks == 3):
			sys.exit("Error checks takes an invalid value!")
		while current_t < tf:
			h = dt
			tolcheck = 0
			self.crashcheck = ode_solver_Nbody(current_t, y_ptr, dy_dt_ptr, main_prop_ptr, semiaxes_ptr, itols, vertices, 
								  face_ids_all, moment_inertia, triggers)
			if self.crashcheck == -1:
				self.docrashcheck = 1
			elif self.crashcheck == -2:
				self.docrashcheck = -1
			
			for i in range(numvars):
				K[0,i] = dy_dt_ptr[i]
			for i in range(n_stages-1):
				#if self.crashcheck >= 0:
				K[i+1] = self.K_step_RK_pointer(y, K, A[i], C[i], i, numvars, current_t, dt, dy_ptr, dy_dt_ptr,
												main_prop_ptr, semiaxes_ptr, itols, vertices, face_ids_all, moment_inertia, triggers)
			y_temp = Step_solution_RK(y, K, B, n_stages, numvars, dt)
			y_tild = Step_solution_RK(y, K, E, n_stages, numvars, dt)
			
			if errorchecks == 3:
				for i in range(numvars):
					localerr += (K[1][i] - K[n_stages-2][i])**2
				localErrorNorm = dt*sqrt(localerr)*err_factor
			else:
				localErrorNorm = Error_adaptive(y, y_temp, y_tild, numvars, atol, rtol, errorchecks)
			if localErrorNorm == 0:
				# There is a problem in some cases in which y_temp = y_tild 
				# when tolerance is too large. This should not be the case as B != E
				# Suspect that double precision is not enough.
				localErrorNorm = 1e-14
				#raise ValueError("Local error is zero. Try lowering ntol, itol or raising quadval inputs.")
			reduce_step = check_reduced_step(localErrorNorm, tolerance, errorchecks)
			if h <= hmin:
				# If step size is smaller than minimum step size, does not reduce it further
				# and sets step size equal to the minimum step size.
				reduce_step = 0
				h = hmin
			while reduce_step == 1:
				h = rescale_stepsize(h, localErrorNorm, fac, p_order_fac, tolerance, hmin, 1.0, errorchecks)
				# Recompute the k values of stage s+1
				for i in range(n_stages-1):
					K[i+1] = self.K_step_RK_pointer(y, K, A[i], C[i], i, numvars, current_t, h, dy_ptr, dy_dt_ptr, 
													main_prop_ptr, semiaxes_ptr, itols, vertices, face_ids_all, moment_inertia, triggers)
				# Step solution for one lower and one higher order
				y_temp = Step_solution_RK(y, K, B, n_stages, numvars, h)
				y_tild = Step_solution_RK(y, K, E, n_stages, numvars, h)
				# Compute the local error
				if errorchecks == 3:
					for i in range(numvars):
						localerr += (K[1][i] - K[n_stages-2][i])**2
					localErrorNorm = dt*sqrt(localerr)*err_factor
				else:
					localErrorNorm = Error_adaptive(y, y_temp, y_tild, numvars, atol, rtol, errorchecks)
				reduce_step = check_reduced_step(localErrorNorm, tolerance, errorchecks)
				tolcheck += 1
				adaptive_hit += 1
				if h <= hmin:
					# If step size is smaller than minimum step size, does not reduce it further
					# and sets step size equal to the minimum step size.
					reduce_step = 0
					h = hmin
			# Once new step size is finished, solution advanced with higher order solution
			if tolcheck > 0:
				dt = h
			else:
				dt = rescale_stepsize(h, localErrorNorm, fac, p_order_fac, tolerance, hmin, hmax, errorchecks)
			for i in range(numvars):
				y_ptr[i] = y_temp[i]
			y = y_temp
			current_t += dt
			step_counter = self.update_arrays(y, current_t, step_counter, SaveFolder)
			if (counter-1) % 10 == 0 and savemid:
				np.savez_compressed(SaveFolder+"MidStepSolution%d" %((counter-1)/10.0), y.base, current_t)
			counter += 1
			if (current_t > progress_checker):
				print("Simulation %.1f%% done... (total time: %.2f s)" %(10*time_counter, time.time()-time_start))
				time_counter += 1
				progress_checker = 0.1*time_counter*tf
			if self.docrashcheck > 0:
				stop_by_collision = ellipsoid_intersect_check_pyx(y, self.semiaxes, triggers[0])
				if stop_by_collision:
					print("Bodies crashed at t = %.5f at step %d" %(current_t, counter))
					self.bodies_crash_fix(counter)
					current_t = tf
				else:
					self.docrashcheck = 0
			elif self.docrashcheck < 0:
				print("Bodies crashed at t = %.5f at step %d" %(current_t, counter))
				self.bodies_crash_fix(counter)
				current_t = tf
		
		if self.NSolFiles > 1:
			self.save_multi_solution(SaveFolder, step_counter)
			
		print("Finished with time %.5f" %(time.time() - time_start))
		print("Number of times time step was lowered: %.d" %adaptive_hit)
		self.nfev = counter
		# Free pointers
		free(y_ptr)
		free(dy_ptr)
		free(dy_dt_ptr)
		free(main_prop_ptr)
		free(vertices)
		free(face_ids_all)
		free(moment_inertia)
		free(semiaxes_ptr)
		if self.NSolFiles > 1:
			self.save_multi_solution(SaveFolder, step_counter-1)
		return 0

	######
	# SET THESE TO FALSE WHEN CODE IS DONE!!!
	######
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef int RK_non_adaptive(self, int n_stages, double[:,:] A, double[:] C, double[:] B, double[:] E, str SaveFolder, int savemid):
		"""
		Steps Runge-Kutta solution without adaptive time stepper.
		No adaptive time stepper are implemented for the dense output methods.

		This function utilizes pointers to send and receive data from the differential equations.
		The pointers are specifically used for C-functions.
		"""
		cdef double[3] itols = [self.iabstol, self.ireltol, self.quadval]
		cdef Py_ssize_t i
		cdef int counter = 1
		cdef int step_counter = 1
		cdef int numvars# = self.NumBodies*12
		cdef int[3] triggers = [self.EulerParams, self.includeSun, self.NumBodies]
		if self.EulerParams:
			numvars = self.NumBodies*13
		else:
			numvars = self.NumBodies*12
		cdef double[:,:] K = np.zeros((n_stages,numvars),dtype=np.float64)
		cdef double t0 = self.tspan[0]
		cdef double tf = self.tspan[self.N_steps-1]
		cdef double dt = (tf-t0)/(self.N_steps_OG-1)
		cdef double current_t = t0
		cdef double[:] y_tild, y_temp
		cdef double[:] y = self.y_init
		# Compute densities
		cdef int N_vertices = 0
		for i in range(self.NumBodies):
			N_vertices += self.p_info[i]*3
		# Pointers to store data used in the C function
		cdef double* y_ptr = <double *>malloc(numvars*sizeof(double))
		cdef double* dy_ptr = <double *>malloc(numvars*sizeof(double))
		cdef double* dy_dt_ptr = <double *>malloc(numvars*sizeof(double))
		cdef double* main_prop_ptr = <double *>malloc((self.NumBodies*5 + 1)*sizeof(double))
		cdef double* semiaxes_ptr = <double *>malloc((self.NumBodies*3)*sizeof(double))
		cdef double* vertices = <double *>malloc(N_vertices*sizeof(double))
		cdef double* moment_inertia = <double *>malloc(self.NumBodies*9*sizeof(double))
		for i in range(numvars):
			y_ptr[i] = y[i]
		for i in range(N_vertices):
			vertices[i] = self.vertices[i]
		for i in range(self.NumBodies):
			semiaxes_ptr[3*i] = self.semiaxes[3*i]
			semiaxes_ptr[3*i+1] = self.semiaxes[3*i+1]
			semiaxes_ptr[3*i+2] = self.semiaxes[3*i+2]
			main_prop_ptr[self.NumBodies*4 + i] = self.p_info[i]
		main_prop_ptr[self.NumBodies*5] = self.G_grav
		cdef int total_faces = 0
		for i in range(self.NumBodies):
			total_faces += self.p_info[self.NumBodies + i]
		cdef int* face_ids_all = <int *>malloc(3*total_faces*sizeof(int))
		for i in range(3*total_faces):
			face_ids_all[i] = self.face_ids[i]

		Sort_mainprop_pointer_polyhedron(main_prop_ptr, vertices, self.NumBodies, self.p_info, self.face_ids, 
										 self.masses, self.semiaxes, moment_inertia)

		cdef double time_counter = 1.0
		cdef double progress_checker = 0.1*time_counter*tf
		self.crashcheck = 0
		self.docrashcheck = 0
		cdef int stop_by_collision = 0
		time_start = time.time()
		while current_t < tf:
			# Compute the k's
			self.crashcheck = ode_solver_Nbody(current_t, y_ptr, dy_dt_ptr, main_prop_ptr, semiaxes_ptr, itols, vertices,
								  face_ids_all, moment_inertia, triggers)
			if self.crashcheck == -1:
				self.docrashcheck = 1
			elif self.crashcheck == -2:
				self.docrashcheck = -1
			for i in range(numvars):
				K[0,i] = dy_dt_ptr[i]
			for i in range(n_stages-1):
				K[i+1] = self.K_step_RK_pointer(y, K, A[i], C[i], i, numvars, current_t, dt, dy_ptr, dy_dt_ptr,
												main_prop_ptr, semiaxes_ptr, itols, vertices, face_ids_all, moment_inertia, triggers)
			# Step solution for one lower and one higher order
			y = Step_solution_RK(y, K, B, n_stages, numvars, dt)
			for i in range(numvars):
				y_ptr[i] = y[i]
			step_counter = self.update_arrays(y, current_t, step_counter, SaveFolder)
			current_t += dt
			self.update_t(current_t, counter)
			if (counter-1) % 10 == 0 and savemid:
				np.savez_compressed(SaveFolder+"MidStepSolution%d" %((counter-1)/10.0), y.base, current_t)
			counter += 1
			if (current_t > progress_checker):
				print("Simulation %.1f%% done... (total time: %.2f s)" %(10*time_counter, time.time()-time_start))
				time_counter += 1
				progress_checker = 0.1*time_counter*tf
			if self.docrashcheck > 0:
				stop_by_collision = ellipsoid_intersect_check_pyx(y, self.semiaxes, triggers[0])
				if stop_by_collision:
					print("Bodies crashed at t = %.5f at step %d" %(current_t, counter))
					self.bodies_crash_fix(counter)
					current_t = tf
				else:
					self.docrashcheck = 0
			elif self.docrashcheck < 0:
				print("Bodies crashed at t = %.5f at step %d" %(current_t, counter))
				self.bodies_crash_fix(counter)
				current_t = tf

		if self.NSolFiles > 1:
			self.save_multi_solution(SaveFolder, step_counter)
			
		print("Finished with time %.5f" %(time.time() - time_start))
		self.nfev = counter
		# Free pointers
		free(y_ptr)
		free(dy_ptr)
		free(dy_dt_ptr)
		free(main_prop_ptr)
		free(vertices)
		free(face_ids_all)
		free(moment_inertia)
		free(semiaxes_ptr)
		return 0