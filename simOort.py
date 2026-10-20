# pylint: disable=C,W
gpu = True
if gpu:
	try:
		import cupy as np
		import cupy.linalg as npl
		import cupy as np
	except:
		pass
import astroUtils as au
from scipy.optimize import newton
import gridUtils as gu
import plotUtils as pu
import time 
import sysUtils as su
import sys
sys.path.insert(1, 'Solvers')
import oortSolver as solver

simName = "oortCloud"
dataDir = "Data/"
N = 256
D = 3
data_drops = 20
cf = .1
L = .1 / 4.
dx = L / N
nf = 1
m22 = np.array([80.0])
rhoDM = 1e7
Mtot = rhoDM * L**3
C = au.G*4*np.pi
Tf = 100.
initial_drop = 0
T_initial = 0
sigma_dm = 216. * au.kms2kpcMyr
n_p = int(1e5)



import cupy as cp

def kepler_to_cartesian(a, e, i, Omega, omega, M, M_sun=1.0):
    """
    Convert Keplerian elements to Cartesian coordinates using CuPy.

    Inputs:
        a      : Semimajor axis [kpc]
        e      : Eccentricity [dimensionless]
        i      : Inclination [radians]
        Omega  : Longitude of ascending node [radians]
        omega  : Argument of perihelion [radians]
        M      : Mean anomaly [radians]
        M_sun  : Solar mass [Msolar]

    Returns:
        r : Positions [kpc], shape (N, 3)
        v : Velocities [kpc/Myr], shape (N, 3)
    """

    G = 4.4985e-12  # kpc^3 / (Msolar Myr^2)
    mu = G * M_sun

    a, e, i, Omega, omega, M = cp.broadcast_arrays(
        cp.asarray(a, dtype=cp.float64),
        cp.asarray(e, dtype=cp.float64),
        cp.asarray(i, dtype=cp.float64),
        cp.asarray(Omega, dtype=cp.float64),
        cp.asarray(omega, dtype=cp.float64),
        cp.asarray(M, dtype=cp.float64)
    )

    # Solve Kepler's equation using Newton iterations
    M = cp.mod(M, 2*cp.pi)

    E = cp.where(e < 0.8, M, cp.pi)

    for _ in range(50):
        f = E - e*cp.sin(E) - M
        fp = 1 - e*cp.cos(E)
        dE = f / fp
        E = E - dE

        if cp.max(cp.abs(dE)).item() < 1e-12:
            break

    # Orbital-plane coordinates
    cosE = cp.cos(E)
    sinE = cp.sin(E)
    sqrt1me2 = cp.sqrt(1 - e**2)

    radius = a * (1 - e*cosE)

    x_orb = a * (cosE - e)
    y_orb = a * sqrt1me2 * sinE

    fac = cp.sqrt(mu*a) / radius

    vx_orb = -fac * sinE
    vy_orb = fac * sqrt1me2 * cosE

    # Rotation matrices
    cO, sO = cp.cos(Omega), cp.sin(Omega)
    co, so = cp.cos(omega), cp.sin(omega)
    ci, si = cp.cos(i), cp.sin(i)

    P = cp.stack([
        cO*co - sO*so*ci,
        sO*co + cO*so*ci,
        so*si
    ], axis=-1)

    Q = cp.stack([
        -cO*so - sO*co*ci,
        -sO*so + cO*co*ci,
        co*si
    ], axis=-1)

    r = x_orb[..., None]*P + y_orb[..., None]*Q
    v = vx_orb[..., None]*P + vy_orb[..., None]*Q

    return r, v


def sample_oort_cloud(
    N,
    gamma=-1.5,
    amin=3000 / 206264806.247,
    amax=100000 / 206264806.247,
    qmin=100 / 206264806.247
):
    """
    Generate Keplerian orbital elements for Oort cloud comets.

    Units:
        a, amin, amax, qmin : kpc
        e, gamma           : dimensionless
        i, Omega, omega, M : radians
    """

    rng = np.random.default_rng()

    if not (0 <= qmin < amax):
        raise ValueError("Require 0 <= qmin < amax")

    def sample_a(n):
        u = rng.uniform(0, 1, n)
        if np.isclose(gamma, -1):
            return amin * (amax / amin)**u
        else:
            g = gamma + 1
            return (amin**g + u * (amax**g - amin**g))**(1/g)

    # Sample a and e, rejecting orbits with q <= qmin
    a = np.empty(N)
    e = np.empty(N)

    remaining = np.arange(N)

    while len(remaining) > 0:
        n = len(remaining)

        a_new = sample_a(n)
        e_new = np.sqrt(rng.uniform(0, 1, n))

        valid = a_new * (1 - e_new) > qmin

        a[remaining[valid]] = a_new[valid]
        e[remaining[valid]] = e_new[valid]

        remaining = remaining[~valid]

    # Isotropic inclination
    cos_i = rng.uniform(-1, 1, N)
    i = np.arccos(cos_i)

    # Orbital orientation
    Omega = rng.uniform(0, 2*np.pi, N)
    omega = rng.uniform(0, 2*np.pi, N)

    # Mean anomaly
    M = rng.uniform(0, 2*np.pi, N)

    return a, e, i, Omega, omega, M


def Boltzmann_distr(s):
	s.psi = np.zeros((nf,N,N,N)) + 0j
	hbar_ = au.h_tilde(m22)[0]
	v2 = s.K*hbar_**2

	# make un-normed version in velocity space
	psi_k = np.exp(-v2 / 2. / sigma_dm**2) + 0j 
	# give random phases
	psi_k *= np.exp(-1j*np.random.uniform(0, 2*np.pi, size = psi_k.shape))
	s.psi[0] = s.GetFFt(psi_k, Forward=False, NoFieldDimension = True)
	s.psi[0,:,:,:] /= np.sqrt(np.sum(np.abs(s.psi[0,:,:,:])**2)*dx**3)
	s.psi[0,:,:,:] *= np.sqrt(Mtot)

def SetICs():
	s = solver.Solver()
	s.dataDir = dataDir

	# sim params
	s.simName = simName
	s.N = N 
	s.np = n_p 
	s.data_drops = data_drops
	s.initial_drop = initial_drop
	s.T_initial = T_initial
	s.cf = cf
	s.gpu = gpu 
	s.eps = 2e-9

	# physics params
	s.L = L 
	s.dx = L / N 
	s.nf = len(m22)
	s.C = C 
	s.Tf = Tf 
	s.m22 = m22 
	s.hbar_ = au.h_tilde(m22)

	a, e, i, Omega, omega, M = sample_oort_cloud(n_p)
	s.r, s.v = kepler_to_cartesian(a, e, i, Omega, omega, M)

	s.r_sun = np.zeros((1,3))
	s.v_sun = np.zeros((1,3))
	s.m_sun = 1.0

	s.set_K()
	Boltzmann_distr(s)

	print(np.max(a), np.min(a), dx)

	return s


def PlotStuff(rho):
	rho = su.cpuThis(rho)
	fo = pu.FigObj()
	fo.AddPlot(rho)
	fo.show()


if __name__ == "__main__":
	# set up sim ics
	s = SetICs()
	PlotStuff(np.abs(s.psi[0,N//2])**2)
	s.RunSim()