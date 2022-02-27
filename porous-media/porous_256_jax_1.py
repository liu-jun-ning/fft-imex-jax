import numpy as np
import jax.numpy as jnp
import jax
from jax import jit,vmap
from jax.config import config
config.update("jax_enable_x64", True) 
#import numexpr as ne
#import matplotlib.pyplot as plt
import time
#from line_profiler import LineProfiler
start=time.time()

def circle(xx, yy, x0, y0):
    return ((xx-x0))**2 + ((yy-y0))**2

def initial_seed(xx, yy, L1, NCom, rho, lmbda2):
	L2 = 256. - L1; kesi = jnp.sqrt(6*lmbda2)
	N = xx.shape[0]
	shape = (NCom, N, N)
	b = 0.5*(jnp.tanh((yy-L2)/kesi)-1)
	Rd = 60.0
	metric = jnp.sqrt(circle(xx, yy, Lx/2, L1))
	phi = jnp.tanh( -(metric-Rd)/kesi )
	phi = jnp.where(yy<=L1,0.,phi)
	phi = jnp.where(yy>=L2,0.5*(jnp.tanh((yy-L2)/kesi)-1),phi)
	return phi

def initial_wall(xx, yy, L1):
	L2 = 256. - L1
	N = xx.shape[0]
	dx = 2.0
	rho1 = jnp.sqrt(3)/2 * ( 1 + jnp.tanh((-yy+L1)/dx) )
	rho = rho1 + jnp.sqrt(3)/2 * ( 1 + jnp.tanh((yy-L2)/dx) )
	return rho

kf = 1; time_init = 0; steps = 60000; dt = 5e-2/kf;
L1 = 40.0; Lx = 256.0; h = 2.0; g=1.0
datafolder = "./"

jfft2 = jax.jit(jnp.fft.fft2)
jifft2 = jax.jit(jnp.fft.ifft2)
def run_simulation(kf, time_init, steps, dt):
	NCom = 1; N = 256; lmbda2 = 1;
	dx = 1.0
	k  = 2 * jnp.pi * jnp.fft.fftfreq(N, dx)
	kx = k.reshape(1,-1)
	ky = k.reshape(-1,1)
	k2 = kx**2 + ky**2
	k4 = k2**2

	x  = jnp.linspace(0, 256, N)[:]
	xx, yy = jnp.meshgrid(x, x)

	rho = initial_wall(xx, yy, L1)
	phi = initial_seed(xx, yy, L1, NCom, rho, lmbda2)
	#phi = np.load(datafolder + "rho_Lrho_big_256_phi_500000.npy")
	#rho = np.load(datafolder + "rho_Lrho_big_256_rho_500000.npy")
	
	phiN = phi.astype(np.complex)
	phiNHat = jfft2(phiN)
	rhoHat = jfft2(rho)
	gradrhoX = jifft2(1j*kx*rhoHat).real
	gradrhoY = jifft2(1j*ky*rhoHat).real
	gradrho2 = 	gradrhoX**2 + gradrhoY**2

	L = (1-rho/jnp.sqrt(3))
	#L = 1.0
	A = 0.5
	Ainv = 1.0/(1+k4*dt*A)

	def body_fun(i, val):	
		phiN, phiNHat = val
		phiNHat = jfft2(phiN)
		mu = phiN * (rho*rho - 1 + phiN*phiN) + 0.5*(h + 0.5*g*phiN) * gradrho2 * lmbda2
		muHat = jfft2(mu) + lmbda2*k2*phiNHat
		xmu = jifft2(1j*kx*muHat)
		ymu = jifft2(1j*ky*muHat)
		xLmuHat = 1j*kx*jfft2(L*xmu)
		yLmuHat = 1j*ky*jfft2(L*ymu)
		temp1 = xLmuHat + yLmuHat
		phiNHat1 = jfft2(phiN)
		phiNHat = (phiNHat1+dt*temp1 + dt*A*k4*phiNHat1)*Ainv
		phiN = jifft2(phiNHat)
		return phiN, phiNHat
	phiN, phiNHat= jax.lax.fori_loop(0, 1000001, body_fun, (phiN, phiNHat))
	jnp.save(datafolder + "rho_Lrho_big_256_jax_phi_1000000.npy", phiN)
	jnp.save(datafolder + "rho_Lrho_big_256_jax_rho_1000000.npy", rho)
'''
lp = LineProfiler()
lp.add_function(circle)
lp.add_function(initial_seed)
lp.add_function(initial_wall)
lp_wrapper = lp(run_simulation)
lp_wrapper(kf, time_init, steps, dt)
lp.print_stats()
run_simulation(kf, time_init, steps, dt)
end=time.time()
timeall=end-start
print(timeall)
'''
run_simulation(kf, time_init, steps, dt)
end=time.time()
timeall=end-start
print(timeall)