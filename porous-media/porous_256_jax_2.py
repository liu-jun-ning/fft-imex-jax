import numpy as np
import jax.numpy as jnp
import jax
from jax import jit,vmap
from jax.config import config
config.update("jax_enable_x64", True) 
#import matplotlib.pyplot as plt
import time
#from line_profiler import LineProfiler
start=time.time()

def circle(xx, yy, x0, y0):
    return ((xx-x0))**2 + ((yy-y0))**2

def initial_seed(xx, yy, L1, NCom, rho, lmbda2):
	L2 = 1.0 - L1; kesi = jnp.sqrt(6*lmbda2)
	N = xx.shape[0]
	shape = (NCom, N, N)
	b = 0.5*(jnp.tanh((yy-L2)/kesi)-1)
	Rd = 30./128
	metric = jnp.sqrt(circle(xx, yy, Lx/2, L1))
	phi = jnp.tanh( -(metric-Rd)/kesi )
	#phi = jax.ops.index_update(phi, jnp.index_exp[:,:],jnp.tanh(-(metric-Rd)/kesi))
	phi = jnp.where(yy<=L1,0.,phi)
	phi = jnp.where(yy>=L2,0.5*(jnp.tanh((yy-L2)/kesi)-1),phi)
	return phi

def initial_wall(xx, yy, L1):
	L2 = 1.0 - L1
	N = xx.shape[0]
	dx = 1.0/N
	rho1 = jnp.sqrt(3)/2 * ( 1 + jnp.tanh((-yy+L1)/dx) )
	rho = rho1 + jnp.sqrt(3)/2 * ( 1 + jnp.tanh((yy-L2)/dx) )
	return rho

kf = 1; time_init = 0; steps = 60000; dt = 2e-6/kf; nplotsteps = 1000
L1 = 0.1; Lx = 1.0; h = 2.0; g=1.0
withGreen = True
datafolder = "./"


jfft2 = jax.jit(jnp.fft.fft2)
jifft2 = jax.jit(jnp.fft.ifft2)
def run_simulation(kf, time_init, steps, dt, nplotsteps, withGreen):
	NCom = 1; N = 256; lmbda2 = 1e-4/2;
	dx = 1.0/N
	k  = 2 * np.pi * np.fft.fftfreq(N, dx)
	kx = k.reshape(1,-1)
	ky = k.reshape(-1,1)
	k2 = kx**2 + ky**2
	k4 = k2**2
	x  = jnp.linspace(0, 1, N)[:]
	xx, yy = jnp.meshgrid(x, x)

	rho = initial_wall(xx, yy, L1)
	phi = initial_seed(xx, yy, L1, NCom, rho, lmbda2)

	phiN = phi.astype(np.complex)
	phiNHat = jfft2(phiN)
	rhoHat = jfft2(rho)
	gradrhoX = jifft2(1j*kx*rhoHat).real
	gradrhoY = jifft2(1j*ky*rhoHat).real
	gradrho2 = 	gradrhoX**2 + gradrhoY**2

	L = (1-rho/np.sqrt(3))
	#L = 1.0
	Ainv = 1.0/(1+k4*dt*L*lmbda2)
	Ainv1 = dt*L*k2


	def body_fun(i, val):	
		phiN, phiNHat = val
		mu = phiN * (rho*rho - 1 + phiN*phiN) + 0.5*(h + 0.5*g*phiN) * gradrho2 * lmbda2
		muHat = jfft2(mu)
		phiNHat1 = Ainv*(phiNHat-Ainv1*muHat)
		phiNHat = phiNHat1
		phiN = jifft2(phiNHat)
		return phiN, phiNHat
	phiN, phiNHat= jax.lax.fori_loop(0, 1000001, body_fun, (phiN, phiNHat))	
	jnp.save(datafolder + "rho_noLrho_jax_256_phi_1000000.npy", phiN)
	#time=63s


'''
lp = LineProfiler()
lp.add_function(circle)
lp.add_function(initial_seed)
lp.add_function(initial_wall)
lp_wrapper = lp(run_simulation)
lp_wrapper(kf, time_init, steps, dt, nplotsteps, withGreen)
lp.print_stats()
'''
run_simulation(kf, time_init, steps, dt, nplotsteps, withGreen)

end=time.time()
timeall=end-start
print(timeall)

