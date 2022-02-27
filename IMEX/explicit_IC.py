import numpy as np

Nx = 128
Ny = 128
np.random.seed(0)
phi0 = (np.random.rand(Nx,Ny)*2 - 1)*0.1
np.save("explicit_IC_phi0.npy",phi0)
print(phi0)