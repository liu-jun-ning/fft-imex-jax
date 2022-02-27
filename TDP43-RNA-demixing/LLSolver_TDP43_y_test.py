import numpy as np

N=128
L=1.0
kx=2*np.pi*np.fft.fftfreq(N, d=L)
ky=2*np.pi*np.fft.fftfreq(N, d=L)
kX,kY=np.meshgrid(kx,ky)
K2=kX**2+kY**2
K4=K2**2

dt=0.001
t=np.arange(0,30000+dt,dt)

A=0.2
eps2=1
lam_P=0.81610
lam_P2=0.408
lam_R=0.098
lam_Y=0.408
X_P2R=4.25
X_P2S=4.25
X_RS=1
X_P2Y=2.5
X_YS=4.25
X_YR=4.25
kon=1
koff=0.01

P0=np.zeros((N,N))
P20=((0.3)*np.random.rand(N,N))
R0=((0.3)*np.random.rand(N,N))
Y0=((0.3)*np.random.rand(N,N))

def simulation(P0,P20,R0,Y0):
    phat=np.fft.fft2(P0)
    phat_i=np.zeros_like(P0)
    phat_i=phat[:,:]
    phat_i_1=np.zeros_like(P0)
    p_i=np.zeros_like(P0)
    p_i[:,:]=P0

    p2hat=np.fft.fft2(P20)
    p2hat_i=np.zeros_like(P20)
    p2hat_i=p2hat[:,:]
    p2hat_i_1=np.zeros_like(P20)
    p2_i=np.zeros_like(P20)
    p2_i[:,:]=P20
    lnp2hat_i=np.zeros_like(P20)
    lnp2hat_i=np.fft.fft2(np.log(p2_i))

    rhat=np.fft.fft2(R0)
    rhat_i=np.zeros_like(R0)
    rhat_i=rhat[:,:]
    rhat_i_1=np.zeros_like(R0)
    r_i=np.zeros_like(R0)
    r_i[:,:]=R0
    lnrhat_i=np.zeros_like(R0)
    lnrhat_i=np.fft.fft2(np.log(r_i))

    yhat=np.fft.fft2(Y0)
    yhat_i=np.zeros_like(Y0)
    yhat_i=yhat[:,:]
    yhat_i_1=np.zeros_like(Y0)
    y_i=np.zeros_like(Y0)
    y_i[:,:]=Y0
    lnyhat_i=np.zeros_like(Y0)
    lnyhat_i=np.fft.fft2(np.log(y_i))

    S0=1-(P20+R0+Y0)
    s_i=np.zeros_like(S0)
    s_i=S0[:,:]
    lnshat_i=np.zeros_like(S0)
    lnshat_i=np.fft.fft2(np.log(s_i))

    phat_deno=1/(1+lam_P*K2*dt)
    p2hat_deno=1/(1+lam_P2*eps2*K4*dt-2*dt*K2*lam_P2*A*X_P2S+koff*dt)
    rhat_deno=1/(1+lam_R*eps2*K4*dt-2*dt*A*lam_R*K2*X_RS)
    yhat_deno=1/(1+lam_Y*eps2*K4*dt-2*dt*A*lam_Y*K2*X_YS)    

    for i in range(len(t)):
        pphat_i=np.fft.fft2(p_i*p_i)
        phat_i_1=(2*koff*p2hat_i*dt+phat_i-2*kon*dt*pphat_i)*phat_deno
        p2hat_i_1=(p2hat_i+kon*dt*pphat_i-dt*K2*A*lam_P2*(lnp2hat_i-lnshat_i+(X_P2Y-X_YS)*yhat_i+(X_P2R-X_RS)*rhat_i+X_P2S*(1-yhat_i-rhat_i)))*p2hat_deno
        rhat_i_1=(rhat_i-A*dt*lam_R*K2*(lnrhat_i-lnshat_i+(X_YR-X_YS)*yhat_i+(X_P2R-X_P2S)*p2hat_i+X_RS*(1-yhat_i-p2hat_i)))*rhat_deno
        yhat_i_1=(yhat_i-A*dt*lam_Y*K2*(lnyhat_i-lnshat_i+(X_P2Y-X_P2S)*p2hat_i+(X_YR-X_RS)*rhat_i+X_YS*(1-p2hat_i-rhat_i)))*yhat_deno
        phat_i=phat_i_1
        p2hat_i=p2hat_i_1
        rhat_i=rhat_i_1
        yhat_i=yhat_i_1
        p_i=np.fft.ifft2(phat_i)
        p2_i=np.fft.ifft2(p2hat_i)
        r_i=np.fft.ifft2(rhat_i)
        y_i=np.fft.ifft2(yhat_i)      
        s_i=1-(p2_i+r_i+y_i)
        lnp2hat_i=np.fft.fft2(np.log(p2_i))
        lnrhat_i=np.fft.fft2(np.log(r_i))
        lnyhat_i=np.fft.fft2(np.log(y_i))
        lnshat_i=np.fft.fft2(np.log(s_i))
        if i%300000==0:
            p_i_ri=p_i.real
            p2_i_ri=p2_i.real
            r_i_ri=r_i.real
            y_i_ri=y_i.real
            np.save('LLSolver_TDP43_y_test333_3ws_%d.npy'%(round(i*dt)),np.vstack((p_i_ri,p2_i_ri,r_i_ri,y_i_ri)))
    return

simulation(P0,P20,R0,Y0)