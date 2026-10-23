# pylint: disable=C,W
import numpy as np_
import os
import time
import sysUtils as su
import mathUtils as mu
import plotUtils as pu
import types
CUPY_IMPORTED = True
import warnings as warn 
try:
	import cupy as cp 
except ImportError:
	CUPY_IMPORTED = False

class Solver():

	def __init__(self, simName = None, N = None, np = None, data_drops = None,
	 initial_drop = 0, T_initial = 0, padded = False, cf = 0.1, gpu = False, eps = 0, L = 1.,
	 dx = None, nf = 1, D = 3, C = 1., Tf = 1., r = [], v = [], dt = 1., v_sun = None,
	 freezeDensity = False, m22 = None, hbar_ = None, psi = [], K = [], r_sun = None,
	 psiSelfGrav = False, m_sun = 1.0):
		
		### simulation parameters
		self.simName = simName
		self.dataDir = "Data/"
		self.N = N
		self.np = np
		self.data_drops = data_drops # int, number of data drops
		self.initial_drop = initial_drop
		self.T_initial = T_initial
		self.padded = padded # bool, pad density with 0s
		self.cf = cf # float, courant factor
		self.gpu = gpu # bool, run on gpu
		self.eps = eps # softening length
		self.freezeDensity = freezeDensity
		self.psiSelfGrav = psiSelfGrav

		### physics parameter
		self.L = L # float, box length
		self.dx = dx # float, pixel size
		self.nf = nf # int, number of fields
		self.D = D # int, number of spatial dimensions
		self.C = C # float, poisson's constant
		self.Tf = Tf # float, final time
		self.m22 = m22 # float, particle mass [1e-22 eV/c^2]
		self.hbar_ = hbar_ # float, hbar / m [kpc^2 / Myr] 

		### dynamical paramters
		self.r = r # array-like, [np, D] positions of particles
		self.r_sun = r_sun # array-like, [1, D] positions of particles
		self.v = v # array-like, [np, D] velocities of particles
		self.v_sun = v_sun # array-like, [np, D] velocities of particles
		self.psi = psi # array-like, [nf,N^D] spatial field
		self.K = K # array-like, [N^3] kinetic update operator argument, kx x kx x kx
		self.dt = dt # float, timestep
		self.m_sun = m_sun

		### diagnostics
		self.times = []
		self.percents = []
		self.counter = 0
		self.timesMax = 100
		self.rho_alias = 0.

		self.extras = {} # dictionary to add to toml


	def OutputToml(self):
		"""
		outputs toml with simulation parameters
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		m22_str = (
		    repr(float(self.m22))
		    if np.ndim(self.m22) == 0
		    else str([float(x) for x in np.asarray(self.m22).ravel()])
		)

		padded_ = "true" if self.padded else "false"
		text = f'''
		# all units in kpc, Msolar, Myr
		[physics]
		Tfinal                      = {self.Tf + self.T_initial} # float, final sim time
		L                           = {self.L} # float, box length
		C 							= {self.C} # float, poisson's constant
		D 							= {self.D} # int, number of spatial dimensions
		m22 						= {m22_str} # m_field

		[simulation]
		N                           = {self.N} # int, grid size
		n_particles                 = {self.np} # int, number of simulation particles
		drops                       = {self.data_drops+ self.initial_drop} # int, number of data drops
		padded 						= {padded_} # bool, pad density with 0s
		c_f                         = {self.cf} # float, timestep courant factor
		'''

		f = open(self.dataDir + f"{self.simName}/meta.toml", "w")
		f.write(text)
		f.close()

		if len(self.extras) > 0:
			extras = {}
			extras['extras'] = self.extras
			su.AddLines2Toml(extras, self.GetTomlString())


	def OutputICs(self):
		"""
		outputs the initial conditions
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		if not(os.path.isdir(self.dataDir + f"{self.simName}/r")) and self.np != None and self.np > 0:
			os.mkdir(self.dataDir + f"{self.simName}/r")
		if not(os.path.isdir(self.dataDir + f"{self.simName}/v")) and self.np != None and self.np > 0:
			os.mkdir(self.dataDir + f"{self.simName}/v")
		if not(os.path.isdir(self.dataDir + f"{self.simName}/psi")):
			os.mkdir(self.dataDir + f"{self.simName}/psi")
		if self.initial_drop != 0:
			self.r_sun_save = np.load(self.dataDir + "r_sun.npy")
		else:
			self.r_sun_save = np.zeros((self.data_drops + 1, 3))
		self.DataDrop(0)

	def set_K(self, K = []):
		"""
		set the spectral gird

		:K: array-like, [N^3] kinetic update operator argument, \n
			i.e. e^{-1j*hbar_*K*dt} is the op \n
			default: use N,L, and D to figure K out
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if len(K) > 0:
			self.K = K 
		else:
			if self.N != None and self.L != None and self.D != None:
				dx = self.L/self.N
				kx = 2*np.pi*np.fft.fftfreq(self.N,d = dx)
				ones = np.ones(self.N)

				self.K = self.get_K(self.N, self.L)
	def get_K(self, N, L):
		"""
		calculate the spectral grid

		:N: int, the grid resolution
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		dx = L/N
		kx = 2*np.pi*np.fft.fftfreq(N, d = dx)
		ones = np.ones(N)

		if self.D == 1:
			return kx**2
		elif self.D == 2:
			K = np.einsum("i,j->ij", kx**2, ones)
			K += np.einsum("i,j->ij", ones, kx**2)
			return K
		elif self.D == 3:
			K = np.einsum("i,j,k->ijk", kx**2, ones, ones)
			K += np.einsum("i,j,k->ijk", ones, kx**2, ones)
			K += np.einsum("i,j,k->ijk", ones, ones, kx**2)
			return K

	def GetFFt(self, psi, Forward = True, NoFieldDimension = False):
		"""
		calculate the fft of psi

		:psi: array-like, [nf, N^D], the fields
		:Forward: bool, forward or backward fft, default: True
		:NoFieldDimension: bool, if True then psi has dim[N^D]

		:return: array-like, fft of psi on all axes except 0th
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if self.D == None:
			self.D = len(psi.shape) - 1

		a = NoFieldDimension

		if Forward:
			if self.D == 1:
				return np.fft.fft(psi, axis = 1 - a)
			elif self.D == 2:
				return np.fft.fft2(psi, axes = (1-a,2-a))
			elif self.D == 3:
				return np.fft.fftn(psi, axes = (1-a,2-a,3-a))
		else:
			if self.D == 1:
				return np.fft.ifft(psi, axis = 1-a)
			elif self.D == 2:
				return np.fft.ifft2(psi, axes = (1-a,2-a))
			elif self.D == 3:
				return np.fft.ifftn(psi, axes = (1-a,2-a,3-a))


	def DataDrop(self, i):
		"""
		output the current status of the dynamical variables
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		if self.np != None and self.np > 0:
			np.save(self.dataDir + self.simName + f"/r/drop{i + self.initial_drop}.npy", self.r)
			np.save(self.dataDir + self.simName + f"/v/drop{i + self.initial_drop}.npy", self.v)
		np.save(self.dataDir + self.simName + f"/m_sun.npy", self.m_sun)
		np.save(self.dataDir + self.simName + f"/psi/drop{i+ self.initial_drop}.npy", self.psi)
		self.r_sun_save[i + self.initial_drop] = self.r_sun[0]
		np.save(self.dataDir + "r_sun.npy", self.r_sun_save)

	def InitializeFiles(self):
		"""
		makes the data directory and outputs the toml file
		"""
		if self.simName == None:
			raise Exception("simName has not been set.\n"+\
				"set simName before initializing files.")

		if not(os.path.isdir(self.dataDir + f"{self.simName}")):
			os.mkdir(self.dataDir + f"{self.simName}")
		self.OutputToml()
		self.OutputICs()

	def get_dt(self, T_remaining, Vmax = None):
		"""
		calculate the timestep

		:T_remaining: float, the time remaining until the next data drop, \n
					default: do not consider this condition
		:Vmax: float, max value of the potential, default: calculate Vmax
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		dt_all = np.zeros(2)

		max_v = np.max(np.abs(self.v))

		dt_all[0] = self.cf * self.dx / max_v
		dt_all[1] = T_remaining
		dt_base = np.min(dt_all)
		
		if self.freezeDensity:
			return dt_base

		k_max = np.sqrt(self.D) * np.pi / self.dx
		delta_k = 2 * k_max / self.N

		dt_kinetic = self.cf * 2. * np.pi / k_max / delta_k / np.max(self.hbar_)
		
		if Vmax == None:
			phi = self.compute_phi()
			phi -= np.mean(phi)
			Vmax = np.max(np.abs(phi))
		dt_potential = self.cf * 2 * np.pi * np.min(self.hbar_) / Vmax
		# rho = self.ComputeParticleDensity()

		dt_array = np.zeros(3)
		dt_array[0] = dt_base
		dt_array[1] = dt_kinetic
		dt_array[2] = dt_potential
		return np.min(dt_array)

	def RunSim(self):
		"""
		run the simulation
		"""
		self.InitializeFiles()
		print("\nrunning simulation " + self.simName + "..." )
		time0 = time.time()
		tNext = float(self.Tf)/self.data_drops
		drop = 1
		T = 0.
		Vmax = None

		while(T < self.Tf):
			T_remaining = tNext - T
			dt = self.get_dt(T_remaining, Vmax=Vmax)
			Vmax = self.Update(dt)
			T += dt 
			self.PrintDiagnostics(T, time0)
			if T_remaining <= 0:
				# su.PrintTimeUpdate(drop,self.data_drops,time0)
				self.DataDrop(drop)
				drop += 1
				tNext = float(self.Tf*drop)/self.data_drops

		if (drop == self.data_drops):
			self.DataDrop(drop)

		su.PrintCompletedTime(time0, "simulation")



	def Update(self, dt):
		"""
		updates the dynamic variables using a drift-kick-drift scheme

		:dt: float, the timestep
		"""
		self.Drift(dt/2.)
		Vmax = self.Kick(dt)
		self.Drift(dt/2.)
		return Vmax


	def Drift(self, dt):
		"""
		update the dynamic variable positions

		:dt: float, timestep
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if len(self.r) > 0 and len(self.v) > 0:
			self.r += self.v * dt
			self.r_sun += self.v_sun * dt 

		if not(self.freezeDensity):
			k2max = (np.pi*self.N/self.L)**2

			self.psi = self.GetFFt(self.psi)

			if self.D == 3:
				self.psi *= np.exp(-1j*dt*\
		            np.einsum("i,jkl->ijkl",self.hbar_, self.K)/(2.))
			elif self.D == 2:
				self.psi *= np.exp(-1j*dt*\
		            np.einsum("i,jk->ijk",self.hbar_, self.K)/(2.))
			elif self.D == 1:
				self.psi *= np.exp(-1j*dt*\
		            np.einsum("i,j->ij",self.hbar_, self.K)/(2.))


			if self.nf == 1:
				if self.D == 3:
					rhoOverThresh = np.sum(np.abs(self.psi[self.K[np.newaxis,:,:,:] > (k2max*.9)])**2)
				elif self.D == 2:
					rhoOverThresh = np.sum(np.abs(self.psi[self.K[np.newaxis,:,:] > (k2max*.9)])**2)
				elif self.D == 1:
					rhoOverThresh = np.sum(np.abs(self.psi[self.K[np.newaxis,:] > (k2max*.9)])**2)
			else:
				rhoOverThresh = 0
			
			rhoToT = np.sum(np.abs(self.psi)**2)
			self.rho_alias = rhoOverThresh / rhoToT

			self.psi = self.GetFFt(self.psi, Forward=False)


	def Kick(self,dt):
		"""
		update the dynamic variable momenta

		:dt: float, timestep
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		phi = self.compute_phi()

		if not(self.freezeDensity) and self.psiSelfGrav:
			if self.D == 3:
				self.psi *= np.exp(-1j*dt*\
	            	np.einsum("i,jkl->ijkl",1./self.hbar_, phi))
			elif self.D == 2:
				self.psi *= np.exp(-1j*dt*\
	            	np.einsum("i,jk->ijk",1./self.hbar_, phi))
			elif self.D == 1:
				self.psi *= np.exp(-1j*dt*\
	            	np.einsum("i,j->ij",1./self.hbar_, phi))

		# calc acceleration on sun
		acc_sun = self.ComputeAcc(phi, self.r_sun)
		self.v_sun += acc_sun * dt

		# calc acceleration on particles
		acc = self.ComputeAcc(phi, self.r)
		acc += self.ExplicitParticleForces()
		self.v += acc * dt

	def GetPeriodicR(self, r_notPeriodic):
		r = r_notPeriodic + self.L / 2.
		r %= self.L 
		r -= self.L / 2.
		return r

	def ExplicitParticleForces(self):
		"""
		compute the explicit acceleration between particles

		:return: array-like, [np, D] acceleration
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp 

		r_ = self.r_sun[0]

		if self.D == 3:
			r_vecs = self.r[:,:] - r_[np.newaxis,:]
			R = np.sqrt(mu.Norm_L2(r_vecs, axes = (1))**2 + self.eps**2 )
			acc = self.C*np.sum(self.r_sun*r_vecs / R[:,np.newaxis]**3, axis = 0) / (4*np.pi)
			return acc

		if self.D == 2:
			r_vecs = self.r[:,:] - r_[np.newaxis,:]
			R = np.sqrt(mu.Norm_L2(r_vecs, axes = (1))**2 + self.eps**2 )
			acc = 2*self.C*np.sum(self.r_sun*r_vecs / R[:,np.newaxis]**2, axis = 0) / (4*np.pi)
			return acc

		if self.D == 1:
			r_vecs = self.r - r_
			acc = self.C*np.sum(self.r_sun*np.sign(r_vecs)) / (4*np.pi)
			return acc

		return acc



	def compute_phi(self):
		"""
		compute the potential
	
		:return: array-like, [N^D]
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		rval = np.sum(np.abs(self.psi)**2, axis = 0)

		rval = self.GetFFt(rval, NoFieldDimension=True)

		K = self.K
		rval = -1*self.C*rval / K
		if self.D == 3:
			rval[0,0,0] = 0.0
		elif self.D ==2:
			rval[0,0] = 0.
		elif self.D == 1:
			rval[0] = 0

		rval = self.GetFFt(rval, Forward = False, NoFieldDimension=True)
		rval = rval.real

		return rval

	def ComputeAcc(self, phi, r):
		"""
		compute the acceleration for the corpuscular particles given a potential

		:phi: array-like, the potential

		:return: array-like, [np, D] acceleration
		"""

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		acc = np.zeros(np.shape(r))

		if len(phi) > 0:
			if self.D == 1:
				acc = self.ComputeAcc1D(phi)
			elif self.D == 2:
				acc = self.ComputeAcc2D(phi)
			elif self.D == 3:
				acc = self.ComputeAcc3D(phi,r)		
		return acc



	def ComputeAcc3D(self, phi, r):
		"""
		compute the 3D acceleration for the corpuscular particles given a potential

		:phi: array-like, the potential

		:return: array-like, [np, D] acceleration
		"""
		N = self.N
		r = self.GetPeriodicR(r)
		dx = self.dx
		L = self.L

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		acc = np.zeros(r.shape)  
		ijk = np.floor((r + L/2.) / dx).astype(int) # [n_particles, [i,j,k]]\
		ijk %= N

		x = dx*(.5+np.arange(-1*N//2, N//2))

		for i in range(self.D):
			axis_ = i 
			gradPhi = mu.gradient_1D(phi, dx, axis_, gpu = self.gpu, padded= self.padded)

			# i,j,k
			f1 = 1 - np.abs(x[ijk[:,0]] - r[:,0])/dx
			f2 = 1 - np.abs(x[ijk[:,1]] - r[:,1])/dx
			f3 = 1 - np.abs(x[ijk[:,2]] - r[:,2])/dx
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i+1,j,k
			f1 = 1. - f1 # dx
			ijk[:,0] += 1
			ijk[:,0] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i+1,j+1,k
			f2 = 1. - f2 # tx
			ijk[:,1] += 1
			ijk[:,1] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i,j+1,k
			f1 = 1. - f1
			ijk[:,0] -= 1
			ijk[:,0] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i,j+1,k+1
			f3 = 1. - f3
			ijk[:,2] += 1
			ijk[:,2] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i,j,k+1
			f2 = 1. - f2
			ijk[:,1] -= 1
			ijk[:,1] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i+1,j,k+1
			f1 = 1. - f1
			ijk[:,0] += 1
			ijk[:,0] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i+1,j+1,k+1
			f2 = 1. - f2
			ijk[:,1] += 1
			ijk[:,1] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			ijk[:,0] -= 1
			ijk[:,1] -= 1
			ijk[:,2] -= 1

		return -1*acc

	def ComputeAcc2D(self, phi):
		"""
		compute the 2D acceleration for the corpuscular particles given a potential

		:phi: array-like, the potential

		:return: array-like, [np, D] acceleration
		"""
		N = self.N
		r = self.r[self.active]
		dx = self.dx
		L = self.L

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		acc = np.zeros(r.shape)  
		ijk = np.floor((r + L/2.) / dx).astype(int) # [n_particles, [i,j,k]]\
		ijk %= N

		x = dx*(.5+np.arange(-1*N//2, N//2))

		for i in range(self.D):
			axis_ = i 
			gradPhi = mu.gradient_1D(phi, dx, axis_, gpu = self.gpu, padded=self.padded)

			# i,j
			f1 = 1 - np.abs(x[ijk[:,0]] - r[:,0])/dx
			f2 = 1 - np.abs(x[ijk[:,1]] - r[:,1])/dx
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1]]*f1*f2

			# i+1,j
			f1 = 1. - f1 # dx
			ijk[:,0] += 1
			ijk[:,0] %= N
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1]]*f1*f2

			# i+1,j+1
			f2 = 1. - f2 # tx
			ijk[:,1] += 1
			ijk[:,1] %= N
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1]]*f1*f2

			# i,j+1
			f1 = 1. - f1
			ijk[:,0] -= 1
			ijk[:,0] %= N
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1]]*f1*f2

			ijk[:,0] -= 1
			ijk[:,1] -= 1

		return -1*acc


	def ComputeAcc1D(self, phi):
		"""
		compute the 1D acceleration for the corpuscular particles given a potential

		:phi: array-like, the potential

		:return: array-like, [np, D] acceleration
		"""
		N = self.N
		r = self.r[self.active]
		dx = self.dx
		L = self.L

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		acc = np.zeros(r.shape)  
		ijk = np.floor((r + L/2.) / dx).astype(int) # [n_particles, [i,j,k]]\
		ijk %= N

		x = dx*(.5+np.arange(-1*N//2, N//2))

		axis_ = 0 
		gradPhi = mu.gradient_1D(phi, dx, axis_, gpu = self.gpu, padded=self.padded)

		# i
		f1 = 1 - np.abs(x[ijk[:,0]] - r[:,0])/dx
		acc[:,0] += gradPhi[ijk[:,0],ijk[:,1]]*f1

		# i+1
		f1 = 1. - f1 # dx
		ijk[:,0] += 1
		ijk[:,0] %= N
		acc[:,0] += gradPhi[ijk[:,0],ijk[:,1]]*f1

		return -1*acc


	def BaseDiagnostics(self, T, time0):
		"""
		prints diagnostic information

		:T: float, sim time completed
		:time0: float, real time the sim started

		:return: string, diagnostic info
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		T_remaining = self.Tf - T
		elapsedTimeTotal = time.time() - time0
		portionDone = T / self.Tf
		portionRemaining = 1 - portionDone

		if self.counter == 0:
			self.times = np.zeros(self.timesMax)
			self.percents = np.zeros(self.timesMax)

		ind_ = self.counter%self.timesMax
		self.times[ind_] = time.time()
		self.percents[ind_] = portionDone

		elapsedTimeRecently = elapsedTimeTotal
		portionDoneRecently = portionDone

		if self.counter == self.timesMax:
			elapsedTimeRecently = \
				(np.max(self.times) - np.min(self.times)) / (len(self.times) - 1)
			portionDoneRecently = \
				(np.max(self.percents) - np.min(self.percents)) / (len(self.percents) - 1)

		self.counter += 1

		string_done = "%.2f done "%(portionDone) 
		string_done += "in %i hrs, %i mins, %i s."%su.hms(elapsedTimeTotal)

		time_estimate = portionRemaining*elapsedTimeRecently / portionDoneRecently
		string_todo = f" eta: %i hrs, %i mins, %i s."%su.hms(time_estimate)
		return string_done + string_todo

	def PrintDiagnostics(self, T, time0):
		"""
		prints diagnostic information

		:T: float, sim time completed
		:time0: float, real time the sim started
		"""

		su.repeat_print(self.BaseDiagnostics(T,time0))