import numpy as np
import matplotlib.pyplot as plt


from mgard import MGARD




#####################################################
# grid

N = 6 * 2**6 + 1

# grid = np.linspace(0,1,N)
grid = np.logspace(0,1,N)
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
mg3 = MGARD(grid, u, order=3)
mg4 = MGARD(grid, u, order=4)


u0 = u.copy()
u0[dind] = mg0.interpolate(ind0, dind, u[ind0])

u1 = u.copy()
u1[dind] = mg1.interpolate(ind0, dind, u[ind0])

u2 = u.copy()
u2[dind] = mg2.interpolate(ind0, dind, u[ind0])

u3 = u.copy()
u3[dind] = mg3.interpolate(ind0, dind, u[ind0])

u4 = u.copy()
u4[dind] = mg4.interpolate(ind0, dind, u[ind0])


plt.subplot(121)
plt.plot(grid, u,  label='u')
plt.step(grid, u0, label='u0')
plt.plot(grid, u1, label='u1')
plt.plot(grid, u2, label='u2')
plt.plot(grid, u3, label='u3')
plt.plot(grid, u4, label='u4')
plt.gca().set_box_aspect(1)
plt.gca().title.set_text('Interpolants')
plt.legend()

plt.subplot(122)
plt.semilogy(grid[dind], np.abs(u-u)[1::2])
plt.semilogy(grid[dind], np.abs(u0-u)[1::2])
plt.semilogy(grid[dind], np.abs(u1-u)[1::2])
plt.semilogy(grid[dind], np.abs(u2-u)[1::2])
plt.semilogy(grid[dind], np.abs(u3-u)[1::2])
plt.semilogy(grid[dind], np.abs(u4-u)[1::2])
plt.gca().set_box_aspect(1)
plt.gca().title.set_text('Errors')

plt.show()