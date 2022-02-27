import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 18})

def p_to_xyz(p):
    s3 = 1/np.sqrt(3.0)
    s6 = 1/np.sqrt(6.0)
    x =  -1*p[0] +  1*p[1] +    0*p[2] +    0*p[3]
    y = -s3*p[0] - s3*p[1] + 2*s3*p[2] +    0*p[3]
    z = -s3*p[0] - s3*p[1] -   s3*p[2] + 3*s6*p[3]
    return x, y, z


def composition_map(c,show=False,fig=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    c0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    x0, y0, z0 = p_to_xyz(c0)
    ax.scatter(x0, y0, z0, c=(c0[1:, :]).reshape(3, -1).T, s=10, alpha=1.0) #[[0 0 0] [1 0 0] [0 1 0] [0 0 1]]黑红绿蓝
    tri = [0, 1, 2, 0, 3, 2, 1, 3]
    ax.plot(x0[tri], y0[tri], z0[tri], 'k-', lw=0.4)
    x, y, z = p_to_xyz(c)
    ax.view_init(20, 52)   #改变立体图视角
    c[c<=0.0] = 0.0
    nCom, nx, ny = c.shape
    n  = nx * ny
    nR = np.random.choice(n, 16384, replace=False)
    x1 = (x.reshape(-1))[nR]
    y1 = (y.reshape(-1))[nR]
    z1 = (z.reshape(-1))[nR]
    colors = (c[1:, :, :]).reshape(3, -1).T
    colors = colors[nR]
    ax.scatter(x1, y1, z1, c=colors, s=1.0, alpha=0.10)
    plt.axis("off")
    if show:
        plt.show()
    plt.close(fig)
    return None

N=128
rootDir = "./"
data=np.load(rootDir+"LLSolver_TDP43_y_test244_3ws_1200.npy")
P2=data[N*1:N*2]
R=data[N*2:N*3]
Y=data[N*3:N*4]
S=1-R-Y-P2
c=np.concatenate((R,P2,Y,S)).reshape(4,N,N)  #c0的黑红绿蓝决定了(R,P2,Y,S)分别为黑红绿蓝
composition_map(c,show=True)