import numpy as np
#import numexpr as ne
#import matplotlib.pyplot as plt
#import time
#from line_profiler import LineProfiler
#start=time.time()

def circle(xx, yy, x0, y0):
    return ((xx-x0))**2 + ((yy-y0))**2

def initial_seed(xx, yy, L1, NCom, rho, lmbda2):
	L2 = 256. - L1; kesi = np.sqrt(6*lmbda2)
	N = xx.shape[0]
	shape = (NCom, N, N)
	phi = 0.0 * np.random.uniform(0.0, 0.05, (N, N))
	b = 0.5*(np.tanh((yy-L2)/kesi)-1)
	Rd = 60.0
	metric = np.sqrt(circle(xx, yy, Lx/2, L1))
	phi[:] = np.tanh( -(metric-Rd)/kesi )
	phi[yy<=L1] = 0;
	y1 = yy[yy>=L2]
	phi[yy>=L2] = 0.5*( np.tanh((y1-L2)/kesi)-1 )
	return phi

def initial_wall(xx, yy, L1):
	L2 = 256. - L1
	N = xx.shape[0]
	dx = 2.0
	rho = np.sqrt(3)/2 * ( 1 + np.tanh((-yy+L1)/dx) )
	rho += np.sqrt(3)/2 * ( 1 + np.tanh((yy-L2)/dx) )
	return rho

steps = 60000; dt = 5e-2;
L1 = 40.0; Lx = 256.0; h = 2.0; g=1.0
datafolder = "./"

def run_simulation(steps, dt):
	NCom = 1; N = 256; lmbda2 = 8;
	dx = 1.0
	k  = 2 * np.pi * np.fft.fftfreq(N, dx)
	kx = k.reshape(1,-1)
	ky = k.reshape(-1,1)
	k2 = kx**2 + ky**2
	k4 = k2**2

	x  = np.linspace(0, 256, N)[:]
	xx, yy = np.meshgrid(x, x)

	rho = initial_wall(xx, yy, L1)
	phi = initial_seed(xx, yy, L1, NCom, rho, lmbda2)
	#phi = np.load(datafolder + "rho_Lrho_big_256_phi_500000.npy")
	#rho = np.load(datafolder + "rho_Lrho_big_256_rho_500000.npy")
	
	phiN = phi.copy()
	rhoHat = np.fft.fft2(rho)
	gradrhoX = np.fft.ifft2(1j*kx*rhoHat).real
	gradrhoY = np.fft.ifft2(1j*ky*rhoHat).real
	gradrho2 = 	gradrhoX**2 + gradrhoY**2

	L = (1-rho/np.sqrt(3))
	#L = 1.0
	A = 0.5
	Ainv = 1.0/(1+k4*dt*A*lmbda2)

	for i in range(1000001):
		phiNHat = np.fft.fft2(phiN)
		mu = phiN * (rho*rho - 1 + phiN*phiN) + 0.5*(h + 0.5*g*phiN) * gradrho2 * lmbda2
		muHat = np.fft.fft2(mu) + lmbda2*k2*phiNHat
		xmu = np.fft.ifft2(1j*kx*muHat)
		ymu = np.fft.ifft2(1j*ky*muHat)
		xLmuHat = 1j*kx*np.fft.fft2(L*xmu)
		yLmuHat = 1j*ky*np.fft.fft2(L*ymu)
		temp1 = xLmuHat + yLmuHat
		phiNHat = np.fft.fft2(phiN)
		phiNHat = (phiNHat+dt*temp1 + dt*A*lmbda2*k4*phiNHat)*Ainv
		phiN = np.fft.ifft2(phiNHat)

		if i % 10000 == 0:
			print(abs(phiN.real).max())
			#imag = (phiN.real).copy()
			#a = plt.imshow(imag, cmap="jet")
			#plt.colorbar(a)
			#plt.savefig(datafolder + "rho_Lrho_big_256_%d.png"%(i))
			np.save(datafolder + "rho_Lrho_big_256_phi_2_%d.npy"%(i), phiN)
			#np.save(datafolder + "rho_Lrho_big_256_rho_2_%d.npy"%(i), rho)
			#plt.close()

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
run_simulation(steps, dt)