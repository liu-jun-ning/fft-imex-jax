import numpy as np
'''
import numexpr as ne
import matplotlib.pyplot as plt
import time
from line_profiler import LineProfiler
start=time.time()
'''

def squicle(xx, yy, x0, y0, a, r):
	p = 2*a/r
	return (abs(xx-x0)/a)**p + (abs(yy-y0)/a)**p


def initial_seed(xx, yy, dR, rho):
	N = xx.shape[0]
	phi = 0.0 * np.random.uniform(0.0, 0.05, (N, N))
	a0 = (1-2*dR)/4
	phi[rho<=0.5] = 1
	for i in range(2):
		for j in range(2):
			x0 = a0 + dR/2 + i * (2*a0 + dR)
			y0 = a0 + dR/2 + j * (2*a0 + dR)
			if (i+j) % 2 == 0:
				r = dR*1.05
			else:
				r = dR*1.0
			metric = squicle(xx, yy, x0, y0, r, r)
			phi[(metric<=1) & (rho<=0.5)] = -1.10
	return phi


dt = 1e-6
h = -2
g = 0.5

def run_simulation(dt):
	NCom = 1; N = 1024; lmbda2 = 2e-6;
	dx = 1.0/N
	k  = 2 * np.pi * np.fft.fftfreq(N, dx)
	kx = k.reshape(-1,1)
	ky = k.reshape(1,-1)
	k2 = kx**2 + ky**2
	k4 = k2**2
	dR = 0.05

	x  = np.linspace(0, 1, N)[:]
	xx, yy = np.meshgrid(x, x)

	rho = 1.0*np.load("3-contact-rho-0.npy")
	phi = 1.0*np.load("3-contact-1000.npy")[0,:,:]

	phiN = phi.copy()
	phiNHat = np.zeros_like(phi, dtype='complex')
	rhoHat = np.fft.fft2(rho)
	gradrhoX = np.fft.ifft2(1j*kx*rhoHat).real
	gradrhoY = np.fft.ifft2(1j*ky*rhoHat).real
	gradrho2 = 	gradrhoX**2 + gradrhoY**2
	
	#L = (1-rho/np.sqrt(3))
	L = 1.0
	A = 0.5*lmbda2
	Ainv = 1.0/(1 + A*k4*dt)

	for i in range(450001):
		phiNHat = np.fft.fft2(phiN)
		mu = phiN * (rho*rho - 1 + phiN*phiN) + 0.5*(h + 0.5*g*phiN) * gradrho2 * lmbda2
		muHat = np.fft.fft2(mu) + lmbda2*k2*phiNHat
		xmu = np.fft.ifft2(1j*kx*muHat)
		ymu = np.fft.ifft2(1j*ky*muHat)
		xLmuHat = 1j*kx*np.fft.fft2(L*xmu)
		yLmuHat = 1j*ky*np.fft.fft2(L*ymu)
		temp1 = xLmuHat + yLmuHat
		phiNHat = (phiNHat+dt*temp1+A*k4*dt*phiNHat)*Ainv
		phiN = np.fft.ifft2(phiNHat)
		if i % 5000 == 0:
			np.save("rho_noLrho_1024_phi_1_%d.npy"%(i), phiN)
			#np.save(datafolder + "rho_noLrho_1024_rho_%d.npy"%(i), rho)
			#print(abs(phiN.real).max())
			#imag = (phiN.real).copy()
			#a = plt.imshow(imag, cmap="jet")
			#plt.colorbar(a)
			#plt.savefig(datafolder + "rho_noLrho_1024_%d.png"%(i))
			#plt.close()

	'''
	Ainv = 1.0/(1 + lmbda2*k4*dt)

	for i in range(2001):
		mu = phiN * (rho*rho - 1 + phiN*phiN) + 0.5*(h + 0.5*g*phiN) * gradrho2 * lmbda2
		muHat = np.fft.fft2(mu)
		xmu = np.fft.ifft2(1j*kx*muHat)
		ymu = np.fft.ifft2(1j*ky*muHat)
		xLmuHat = 1j*kx*np.fft.fft2(L*xmu)
		yLmuHat = 1j*ky*np.fft.fft2(L*ymu)
		temp1 = xLmuHat + yLmuHat
		phiNHat = (phiNHat+dt*temp1)*Ainv
		phiN = np.fft.ifft2(phiNHat)

		if (i+1) % 1000 == 0:
			print(abs(phiN.real).max())
			imag = (phiN.real).copy()
			a = plt.imshow(imag, cmap="jet")
			plt.colorbar(a)
			plt.savefig(datafolder + "rho_noLrho_1024_%d.png"%(i))
			plt.close()
	'''

run_simulation(dt)