import numpy as np
from matplotlib import pyplot as plt

a = np.arange(0,995000,5000)
N = 128
#a=np.array([0])
A = np.array([0,0.0625,0.125,0.25,0.5,1])
error = np.zeros_like(A)

for j in range(len(A)):
	b3 = np.zeros((N,N))
	for i in range(len(a)):
		#b1 = np.load("error_explicit/explicit_phi_lam1_%d.npy"%(a[i])) - np.load("error_semi/error_semi_phi_lam1_A1_A%d_%d.npy"%(j,a[i]))
		#c1 = abs(b1).max()
		#if error[j]<=c1:
			#error[j]=c1
		b1 = np.load("error_explicit/explicit_phi_lam0.01_%d.npy"%(a[i])) - np.load("error_semi/error_semi_phi_lam0.01_A1_A%d_%d.npy"%(j,a[i]))
		b2 =  b1**2
		b3 = b2 + b3
		if error[j] <= abs(b3).max()/len(a):
			error[j] = abs(b3).max()/len(a)

'''
error0 = 0
for i in range(len(a)):
	b2 = np.load("error_explicit/error_exact_phi_%d.npy"%(a[i]*1000)) - np.load("error_semi/error_semi_phi_dt0.001_A0.00005_%d.npy"%a[i])
	c2 = abs(b2).max()
	if error0<=c2:
		error0 = c2

A = np.insert(A,6,0.00005)
error = np.insert(error,6,error0)
'''

print(error)
np.save("error_semi_phi_lam0.01_dt_3_995000.npy",error)
plt.figure(figsize=(14,6),dpi=80)
plt.plot(A,error,label='lam2=0.01',color='blue')
#plt.xlim((0,0.06))
plt.xlabel('A')
plt.ylabel('error_semi')
plt.legend()
plt.show()