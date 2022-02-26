import numpy as np
#import numexpr as ne
import jax.numpy as jnp
import jax
from jax import jit,vmap
from jax.config import config
config.update("jax_enable_x64", True) 
import matplotlib.pyplot as plt
import time
from line_profiler import LineProfiler
start=time.time()

def squicle(xx, yy, x0, y0, a, r):
	p = 2*a/r
	return (abs(xx-x0)/a)**p + (abs(yy-y0)/a)**p


def initial_seed(xx, yy, dR, NCom, rho):
	N = xx.shape[0]
	shape = (NCom, N, N)
	phi = jnp.zeros((N, N))
	a0 = (1-2*dR)/4
	phi = jnp.where(rho<=0.5,1.,phi)
	for i in range(2):
		for j in range(2):
			x0 = a0 + dR/2 + i * (2*a0 + dR)
			y0 = a0 + dR/2 + j * (2*a0 + dR)
			if (i+j) % 2 == 0:
				r = dR*1.05
			else:
				r = dR*1.0
			metric = squicle(xx, yy, x0, y0, r, r)
			phi = jnp.where((metric<=1) & (rho<=0.5),-1.10,phi)
	return phi


def initial_wall(xx, yy, dR):
	rho = 0.0 * xx
	a = (1-2*dR)/4

	for i in range(3):
		for j in range(3):
			x0 = i * (2*a+dR)
			y0 = j * (2*a+dR)
			metric = squicle(xx, yy, x0, y0, a, a/2)
			rho = jnp.where(metric<=1,jnp.sqrt(3),rho)
	return rho

time_init = 0; steps = 1000000; dt = 1e-6; nplotsteps = 1000;
h = -2; g = 0.5;
datafolder = "./"
jfft2 = jax.jit(jnp.fft.fft2)
jifft2 = jax.jit(jnp.fft.ifft2)

def run_simulation(time_init, steps, dt, nplotsteps):
	NCom = 1; N = 1024; lmbda2 = 2e-6;
	dx = 1.0/N
	k  = 2 * jnp.pi * jnp.fft.fftfreq(N, dx)
	kx = k.reshape(-1,1)
	ky = k.reshape(1,-1)
	k2 = kx**2 + ky**2
	k4 = k2**2
	dR = 0.05

	x  = jnp.linspace(0, 1, N)[:]
	xx, yy = jnp.meshgrid(x, x)

	#rho = initial_wall(xx, yy, dR)
	rho = 1.0*jnp.load(datafolder + "3-contact-rho-%d.npy"% (time_init))
	phi = initial_seed(xx, yy, dR, NCom, rho)

	phiN = phi.astype(np.complex)
	phiNHat = jnp.zeros_like(phi, dtype='complex')
	rhoHat = jfft2(rho)
	gradrhoX = jifft2(1j*kx*rhoHat).real
	gradrhoY = jifft2(1j*ky*rhoHat).real
	gradrho2 = 	gradrhoX**2 + gradrhoY**2
	
	L = (1-rho/jnp.sqrt(3))
	#L = 1.0
	A = 0.5
	Ainv = 1.0/(1 + A*k4*dt)

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
		phiNHat = jfft2(phiN)
		phiNHat1 = (phiNHat+dt*temp1+A*k4*dt*phiNHat)*Ainv
		phiNHat = phiNHat1
		phiN = jifft2(phiNHat)
		return phiN, phiNHat
	phiN, phiNHat= jax.lax.fori_loop(0, 10001, body_fun, (phiN, phiNHat))
	print(abs(phiN.real).max())
	imag = phiN.real
	a = plt.imshow(imag, cmap="jet")
	plt.colorbar(a)
	plt.savefig(datafolder + "rho_noLrho_1024_%d.png"%(i))
	plt.close()

lp = LineProfiler()
lp.add_function(squicle)
lp.add_function(initial_seed)
lp.add_function(initial_wall)
lp_wrapper = lp(run_simulation)
lp_wrapper(time_init, steps, dt, nplotsteps)
lp.print_stats()
run_simulation(time_init, steps, dt, nplotsteps)
end=time.time()
timeall=end-start
print(timeall)