import numpy as np
import os
import time
import sys

rundate = time.strftime("%Y/%m_%d/")
Folder_location = os.path.dirname(os.path.realpath(__file__)) + "/OutputData/"

class ToScipyClass:
	def __init__(self, numpysol, N_bodies, numfiles, use_eulerP, simplify=True, file_exist=True):
		""" Convert saved solution numpy file to a sol class instance """
		self.N_bodies = N_bodies
		self.NSolFiles = numfiles
		self.use_euler_params = use_eulerP
		if file_exist:
			self.file_exist = True
			if simplify:
				self.simplify_arrays(numpysol)
			else:
				self.t = numpysol['arr_0']
				self.var = numpysol['arr_1']
				self.nfev = numpysol['arr_2']
		else:
			self.file_exist = False
			self.t = np.zeros(0)
			self.var = np.zeros(0)
			self.nfev = 0
			
	def simplify_arrays(self, NpSol):
		"""
		A class that separates the solution variables so it is easier to access.
		The scipy solution instance has two parameters: sol.t and sol.var, where
		sol.var contains solution for all physical parameters for all bodies.
		This class separates these parameters so that
		sol.X = [x_pos1, x_pos2, ...]
		sol.X = [y_pos1, y_pos2, ...] etc.
		"""
		self.t = NpSol['arr_0']
		yvals = NpSol['arr_1']
		self.nfev = NpSol['arr_2']
		NB3 = self.N_bodies*3
		self.vx = yvals[0:NB3:3]
		self.vy = yvals[1:NB3+1:3]
		self.vz = yvals[2:NB3+2:3]
		self.x = yvals[NB3:2*NB3:3]
		self.y = yvals[NB3+1:2*NB3+1:3]
		self.z = yvals[NB3+2:2*NB3+2:3]
		self.omegax = yvals[2*NB3:3*NB3:3]
		self.omegay = yvals[2*NB3+1:3*NB3+1:3]
		self.omegaz = yvals[2*NB3+2:3*NB3+2:3]
		if self.use_euler_params:
			self.e0 = yvals[3*NB3:4*NB3:4]
			self.e1 = yvals[3*NB3+1:4*NB3+1:4]
			self.e2 = yvals[3*NB3+2:4*NB3+2:4]
			self.e3 = yvals[3*NB3+3:4*NB3+3:4]
		else:
			self.phi = yvals[3*NB3:4*NB3:3]
			self.theta = yvals[3*NB3+1:4*NB3+1:3]
			self.psi = yvals[3*NB3+2:4*NB3+2:3]

	def info(self):
		""" Prints information of the data """
		print("===========================")
		print("---Global info:")
		print("Number of bodies: %d" %self.N_bodies)
		print("Number of solution files: %d" %self.NSolFiles)
		print("Number of evaluations: %d" %self.nfev)
		print("---Angle info:")
		if self.use_euler_params:
			print("Euler parameters are considered in this data set")
			print("Use parameters: e0, e1, e2, e3")
		else:
			print("Tait-Bryan are considered in this data set")
			print("Use angles: phi, theta, psi")
		print("===========================")
			
def save_solution(data_output_dir, sol, elparams, masses, outfile, use_eulerP):
	""" 
	Saves solution file into a numpy file 
	Also saves ellipsoid parameters and masses
	"""
	if not os.path.isdir(data_output_dir):
		os.makedirs(data_output_dir)
	np.savez_compressed(outfile, sol.t, sol.var, sol.nfev, elparams, masses, sol.NSolFiles, use_eulerP)

def read_solution(foldname, rundate=rundate, simplify=True, foldloc=Folder_location, extension="", skipread=False, ignorewarn=False):
	""" 
	Reads solution file from a numpy file 
	Only takes folder name of saved data. Can specify folder location and rundate.
	"""
	outfolder = foldloc + rundate + foldname
	path, dirs, files = next(os.walk(outfolder))
	if extension:
		outfile = foldloc + rundate + foldname + "/ODE_solution%s.npz" %extension
	else:
		outfile = foldloc + rundate + foldname + "/ODE_solution.npz"
	if not os.path.isfile(outfile):
		if skipread:
			if not ignorewarn:
				print("File %s does not exist. Skipping" %(outfile))	
			sol = ToScipyClass(0, 0, 0, 0, simplify=simplify, file_exist=False)
			return sol, 0, 0
		else:
			sys.exit("File %s does not exist." %(outfile))
	else:
		npfile = np.load(outfile)
		elparams = npfile['arr_3']
		masses = npfile['arr_4']
		numfiles = npfile['arr_5']
		use_eulerP = npfile['arr_6']

		sol = ToScipyClass(npfile, len(masses), numfiles, use_eulerP, simplify=simplify)
	return sol, elparams, masses

def save_params(params, outfile):
	""" Saves other parameters, e.g. energies """
	np.savez_compressed(outfile, params)

def read_params(foldname, rundate=rundate, foldloc=Folder_location):
	""" Reads energies. """
	outfile = foldloc + rundate + foldname + "/energies.npz"
	elparams = np.load(outfile)
	return elparams['arr_0']

class ReadMultiSolutions():
	"""
	If the simulations have too many points, the solution files are split into multiple parts.
	This is to prevent the program to use all the memory during the simulations.
	"""
	def __init__(self, foldname, index, N_bodies, use_eulerP, rundate=rundate, simplify=True, foldloc=Folder_location, mergeall=False):
		outfolder = foldloc + rundate + foldname
		path, dirs, files = next(os.walk(outfolder))
		self.numfiles = 0
		for i in range(len(files)):
			if "Solutions" in files[i]:
				self.numfiles += 1
		self.outfolder = outfolder
		self.N_bodies = N_bodies
		self.use_eulerP = use_eulerP
		self.read_multisol(index, mergeall=mergeall, simplify=simplify)

	def read_multisol(self, index, mergeall=False, simplify=False):
		"""
		Reads the separated solution files if the number of points exceed maximum array size.
		If mergeall = True, merges all solution files into one class instance.
		Not recommended for multibody simulations as well as long simulation times.

		Variable 'index' is the i'th solution file. For example, if there are following files:
		Solutions1.npz
		Solutions2.npz
		Solutions3.npz
		Then index=2 choses Solutions2.npz.	
		"""
		
		if mergeall:
			outfile = self.outfolder + "/Solutions1.npz"
			npfile = np.load(outfile)
			self.t = npfile['arr_0']
			mega_arr = npfile['arr_1']
			vx = mega_arr[0].transpose()
			vy = mega_arr[1].transpose()
			vz = mega_arr[2].transpose()
			x = mega_arr[3].transpose()
			y = mega_arr[4].transpose()
			z = mega_arr[5].transpose()
			omegax = mega_arr[6].transpose()
			omegay = mega_arr[7].transpose()
			omegaz = mega_arr[8].transpose()
			if self.use_eulerP:
				e0 = mega_arr[9].transpose()
				e1 = mega_arr[10].transpose()
				e2 = mega_arr[11].transpose()
				e3 = mega_arr[12].transpose()
			else:
				phi = mega_arr[9].transpose()
				theta = mega_arr[10].transpose()
				psi = mega_arr[11].transpose()
			for i in range(2,self.numfiles+1):
				outfile = self.outfolder + "/Solutions%d.npz" %i
				npfile = np.load(outfile)
				self.t = np.concatenate([self.t, npfile['arr_0']])
				mega_arr = npfile['arr_1']
				vx = np.hstack([vx, mega_arr[0].transpose()])
				vy = np.hstack([vy, mega_arr[1].transpose()])
				vz = np.hstack([vz, mega_arr[2].transpose()])
				x = np.hstack([x, mega_arr[3].transpose()])
				y = np.hstack([y, mega_arr[4].transpose()])
				z = np.hstack([z, mega_arr[5].transpose()])
				omegax = np.hstack([omegax, mega_arr[6].transpose()])
				omegay = np.hstack([omegay, mega_arr[7].transpose()])
				omegaz = np.hstack([omegaz, mega_arr[8].transpose()])
				if self.use_eulerP:
					e0 = np.hstack([e0, mega_arr[9].transpose()])
					e1 = np.hstack([e1, mega_arr[9].transpose()])
					e2 = np.hstack([e2, mega_arr[9].transpose()])
					e3 = np.hstack([e3, mega_arr[9].transpose()])
				else:
					phi = np.hstack([phi, mega_arr[9].transpose()])
					theta = np.hstack([theta, mega_arr[10].transpose()])
					psi = np.hstack([psi, mega_arr[11].transpose()])
		else:
			if index < 1:
				raise ValueError("Index for multifile solution reading must be greater than zero.")
			if index > self.numfiles:
				raise ValueError("Index for multifile solution reading cannot be greater than number of files.")
			outfile = self.outfolder + "/Solutions%d.npz" %index
			npfile = np.load(outfile)
			self.t = npfile['arr_0']
			mega_arr = npfile['arr_1']
			vx = mega_arr[0].transpose()
			vy = mega_arr[1].transpose()
			vz = mega_arr[2].transpose()
			x = mega_arr[3].transpose()
			y = mega_arr[4].transpose()
			z = mega_arr[5].transpose()
			omegax = mega_arr[6].transpose()
			omegay = mega_arr[7].transpose()
			omegaz = mega_arr[8].transpose()
			if self.use_eulerP:
				e0 = mega_arr[9].transpose()
				e1 = mega_arr[10].transpose()
				e2 = mega_arr[11].transpose()
				e3 = mega_arr[12].transpose()
			else:
				phi = mega_arr[9].transpose()
				theta = mega_arr[10].transpose()
				psi = mega_arr[11].transpose()
		if not simplify:
			pos_idx = self.N_bodies*3
			angvel_idx = self.N_bodies*6
			ang_idx = self.N_bodies*9
			self.var = np.zeros((12*self.N_bodies,len(self.t)))
			for i in range(self.N_bodies):
				C_i = 3*i
				self.var[C_i + 0] = vx[i]
				self.var[C_i + 1] = vy[i]
				self.var[C_i + 2] = vz[i]
				self.var[pos_idx + C_i + 0] = x[i]
				self.var[pos_idx + C_i + 1] = y[i]
				self.var[pos_idx + C_i + 2] = z[i]
				self.var[angvel_idx + C_i + 0] = omegax[i]
				self.var[angvel_idx + C_i + 1] = omegay[i]
				self.var[angvel_idx + C_i + 2] = omegaz[i]
				if self.use_eulerP:
					D_i = 4*i
					self.var[ang_idx + D_i + 0] = e0[i]
					self.var[ang_idx + D_i + 1] = e1[i]
					self.var[ang_idx + D_i + 2] = e2[i]
					self.var[ang_idx + D_i + 3] = e3[i]
				else:
					self.var[ang_idx + C_i + 0] = phi[i]
					self.var[ang_idx + C_i + 1] = theta[i]
					self.var[ang_idx + C_i + 2] = psi[i]
			
			#self.y = np.array([vx, vy, vz, X, Y, Z, omegax, omegay, omegaz, phi, theta, psi])
		else:
			self.vx = vx
			self.vy = vy
			self.vz = vz
			self.x = x
			self.y = y
			self.z = z
			self.omegax = omegax
			self.omegay = omegay
			self.omegaz = omegaz
			if self.use_eulerP:
				self.e0 = e0
				self.e1 = e1
				self.e2 = e2
				self.e3 = e3
			else:
				self.phi = phi
				self.theta = theta
				self.psi = psi
		self.nfev = len(self.t)

	def info(self):
		""" Prints information of the data """
		print("===========================")
		print("---Global info:")
		print("Number of bodies: %d" %self.N_bodies)
		print("Number of evaluations for this solution file: %d" %self.nfev)
		print("---Angle info:")
		if self.use_eulerP:
			print("Euler parameters are considered in this data set")
			print("Use parameters: e0, e1, e2, e3")
		else:
			print("Tait-Bryan are considered in this data set")
			print("Use angles: phi, theta, psi")
		print("===========================")

def save_basic_format(data_output_dir, sol, energies, outfile, outfile_energy, use_eulerP, Nbodies, iscsv=False):
	"""
	Converts Numpy solution files to a .txt or .bin file.
	"""
	import other_functions as OF
	if not os.path.isdir(data_output_dir):
		os.makedirs(data_output_dir)
	N3 = 3*Nbodies
	N6 = 6*Nbodies
	N9 = 9*Nbodies
	if iscsv:
		import csv
		with open(outfile, "w", newline='') as text_file:
			outputwrite = csv.writer(text_file)
			for i in range(len(sol.t)):
				into_csv = []
				into_csv.append("%.6f " %sol.t[i])
				for j in range(Nbodies):
					C_j = 3*j
					D_j = 4*j
					into_csv.append("%.16f " %sol.var[N3+C_j][i])
					into_csv.append("%.16f " %sol.var[N3+C_j+1][i])
					into_csv.append("%.16f " %sol.var[N3+C_j+2][i])
				for j in range(Nbodies):
					C_j = 3*j
					D_j = 4*j
					into_csv.append("%.16f " %sol.var[C_j][i])
					into_csv.append("%.16f " %sol.var[C_j+1][i])
					into_csv.append("%.16f " %sol.var[C_j+2][i])
				for j in range(Nbodies):
					C_j = 3*j
					D_j = 4*j
					into_csv.append("%.16f " %sol.var[N6+C_j][i])
					into_csv.append("%.16f " %sol.var[N6+C_j+1][i])
					into_csv.append("%.16f " %sol.var[N6+C_j+2][i])
				for j in range(Nbodies):
					C_j = 3*j
					D_j = 4*j
					if use_eulerP:
						R = OF.rotation_matrix_euler(sol.var[N9+D_j][i], sol.var[N9+D_j+1][i], sol.var[N9+D_j+2][i], sol.var[N9+D_j+3][i])
					else:
						R = OF.rotation_matrix(sol.var[N9+C_j][i], sol.var[N9+C_j+1][i], sol.var[N9+C_j+2][i])
					into_csv.append("%.16f " %R[0][0])
					into_csv.append("%.16f " %R[0][1])
					into_csv.append("%.16f " %R[0][2])
					into_csv.append("%.16f " %R[1][0])
					into_csv.append("%.16f " %R[1][1])
					into_csv.append("%.16f " %R[1][2])
					into_csv.append("%.16f " %R[2][0])
					into_csv.append("%.16f " %R[2][1])
					into_csv.append("%.16f " %R[2][2])
				outputwrite.writerow(into_csv)
		with open(outfile_energy, "w", newline='') as text_file:
			outputwrite = csv.writer(text_file)
			for i in range(len(sol.t)):
				outputwrite.write(["%.16f", "%.16f", "%.16f"] %(energies[0][i], energies[1][i], energies[2][i]))
	else:
		with open(outfile, "w") as text_file:
			for i in range(len(sol.t)):
				text_file.write("%.6f " %sol.t[i])
				for j in range(Nbodies):
					C_j = 3*j
					D_j = 4*j
					text_file.write("%.16f " %sol.var[N3+C_j][i])
					text_file.write("%.16f " %sol.var[N3+C_j+1][i])
					text_file.write("%.16f " %sol.var[N3+C_j+2][i])
				for j in range(Nbodies):
					C_j = 3*j
					D_j = 4*j
					text_file.write("%.16f " %sol.var[C_j][i])
					text_file.write("%.16f " %sol.var[C_j+1][i])
					text_file.write("%.16f " %sol.var[C_j+2][i])
				for j in range(Nbodies):
					C_j = 3*j
					D_j = 4*j
					text_file.write("%.16f " %sol.var[N6+C_j][i])
					text_file.write("%.16f " %sol.var[N6+C_j+1][i])
					text_file.write("%.16f " %sol.var[N6+C_j+2][i])
				for j in range(Nbodies):
					C_j = 3*j
					D_j = 4*j
					if use_eulerP:
						R = OF.rotation_matrix_euler(sol.var[N9+D_j][i], sol.var[N9+D_j+1][i], sol.var[N9+D_j+2][i], sol.var[N9+D_j+3][i])
					else:
						R = OF.rotation_matrix(sol.var[N9+C_j][i], sol.var[N9+C_j+1][i], sol.var[N9+C_j+2][i])
					text_file.write("%.16f " %R[0][0])
					text_file.write("%.16f " %R[0][1])
					text_file.write("%.16f " %R[0][2])
					text_file.write("%.16f " %R[1][0])
					text_file.write("%.16f " %R[1][1])
					text_file.write("%.16f " %R[1][2])
					text_file.write("%.16f " %R[2][0])
					text_file.write("%.16f " %R[2][1])
					text_file.write("%.16f " %R[2][2])
				text_file.write("\n")
		with open(outfile_energy, "w") as text_file:
			for i in range(len(sol.t)):
				text_file.write("%.16f %.16f %.16f \n" %(energies[0][i], energies[1][i], energies[2][i]))
				


if __name__ == '__main__':
	## For testing purposes
	
	#folderloc = 'C:/Users/alexho/Documents/PhD_stuff/Spheroid_code/OutputData/2019/10_15/init_input/'
	#sol = read_midstep_solution(folderloc, interpolate=True, NInterpFac=10)
	#sol,elparams,masses = read_solution("init_input", simplify=True)
	#print(sol.N_bodies)
	#read_multisol("init_input")
	sol = ReadMultiSolutions("init_input", rundate="2019/12_18/")
	sol.read_multisol(1, mergeall=True, simplify=False)
	