import numpy as np
import matplotlib.pyplot as plt


from mgard import MGARD




#####################################################
# grid

N = 2**4 + 1
# N = 2**5 + 1
# N = 2**10 + 1

grid = np.linspace(0,1,N)
ind0 = np.arange(0,N,2)
dind = np.arange(1,N,2)


#####################################################
# data

u = np.sin(np.arange(N)/N*2*np.pi)
# u = (np.arange(N).astype(float))**3

# # random data
# u = np.zeros_like(grid)
# for i in range(1,50):
# 	u += np.random.randn()/i * np.sin(np.arange(N)/N*i*np.pi)
# u -= u.mean()



########################
mg0 = MGARD(grid, u, order=0)
mg1 = MGARD(grid, u, order=1)
mg2 = MGARD(grid, u, order=2)


mg0.decompose_full()
mg1.decompose_full()
mg2.decompose_full()

mg0.recompose_full()
mg1.recompose_full()
mg2.recompose_full()

plt.plot(np.abs(u-mg0.u_mg), label='u0')
plt.plot(np.abs(u-mg1.u_mg), label='u1')
plt.plot(np.abs(u-mg2.u_mg), label='u2')

plt.legend()
plt.show()