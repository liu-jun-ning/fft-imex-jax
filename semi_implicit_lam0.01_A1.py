import numpy as np
#from matplotlib import pyplot as plt

dt = 1e-4
t = np.arange(0,100,dt)
lam2 = 0.01
A = np.array([0,0.0625,0.125,0.25,0.5,1])

dx = 1.0
dy = 1.0
N = 128
Nx = N
Ny = N
kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
kX,kY = np.meshgrid(kx,ky)      
K2 = kX**2 + kY**2
K4=K2**2

phi0 = np.zeros((len(A),Nx,Ny))

def simulation_exact(phi0):
	phi0hat = np.zeros_like(phi0)
	phi_i = np.zeros_like(phi0)
	phi_i_1 = np.zeros_like(phi0)
	phihat_i = np.zeros_like(phi0)
	phihat_i_1 = np.zeros_like(phi0)
	phi3_i = np.zeros_like(phi0)
	phi3hat = np.zeros_like(phi0)
	for j in range(len(A)):
		phi0[j,:,:] = np.load("explicit_IC_phi0.npy")
		phi0hat[j,:,:] = np.fft.fft2(phi0[j,:,:])
		phi_i = phi0
		phihat_i = phi0hat
		den = 1/(1-dt+lam2*K4*dt+A[j]*K4*dt)
		for i in range(len(t)):
			phi3_i[j,:,:]=phi_i[j,:,:]**3
			phi3hat[j,:,:]=np.fft.fft2(phi3_i[j,:,:])
			phihat_i_1[j,:,:] = den*(phihat_i[j,:,:]-dt*phi3hat[j,:,:]+A[j]*K4*dt*phihat_i[j,:,:])
			phihat_i[:,:,:] = phihat_i_1
			phi_i[j,:,:] = np.fft.ifft2(phihat_i[j,:,:])
			if i % 5000 == 0:
				np.save("error_semi_phi_lam0.01_A1_A%d_%d.npy"%(j,i),phi_i[j,:,:].real)
	return 

simulation_exact(phi0)