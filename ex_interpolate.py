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


u0 = u.copy()
u0[dind] = mg0.interpolate(ind0, dind, u[ind0])

u1 = u.copy()
u1[dind] = mg1.interpolate(ind0, dind, u[ind0])

u2 = u.copy()
u2[dind] = mg2.interpolate(ind0, dind, u[ind0])


plt.plot(u,  label='u')
# plt.plot(u0, label='u0')
plt.plot(u1, label='u1')
plt.plot(u2, label='u2')

plt.legend()
plt.show()