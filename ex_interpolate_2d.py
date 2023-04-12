import numpy as np
import matplotlib.pyplot as plt


from mgard import MGARD




#####################################################
# grid

# N = 6 * 2**1 + 1
N = 2**5 + 1

dim = 2

grid = [np.linspace(0,1,N) for _ in range(dim)]
ind  = [np.arange(0,N)     for _ in range(dim)]
ind0 = [np.arange(0,N,2)   for _ in range(dim)]
dind = [np.arange(1,N,2)   for _ in range(dim)]


#####################################################
# data


u = np.cos(np.arange(N)/N*2*np.pi)
for d in range(1,dim):
	v = np.cos(np.arange(N)/N*2*np.pi)
	u = u[...,np.newaxis] * v.reshape([1]*(d-1)+[len(v)])

# # random data
# u = np.zeros_like(grid)
# for i in range(1,50):
# 	u += np.random.randn()/i * np.sin(np.arange(N)/N*i*np.pi)
# u -= u.mean()


########################
# mg0 = MGARD(grid, u, order=0)
mg1 = MGARD(grid, u, order=(1,1))
# mg2 = MGARD(grid, u, order=2)
# mg3 = MGARD(grid, u, order=3)
# mg4 = MGARD(grid, u, order=4)


# u0 = u.copy()
# u0[dind] = mg0.interpolate(ind0, dind, u[ind0])

u1 = np.zeros_like(u)
u1[np.ix_(*ind0)] = u[np.ix_(*ind0)]
mg1.interpolate_nd(ind, ind0, dind)

# u2 = u.copy()
# u2[dind] = mg2.interpolate(ind0, dind, u[ind0])

# u3 = u.copy()
# u3[dind] = mg3.interpolate(ind0, dind, u[ind0])

# u4 = u.copy()
# u4[dind] = mg4.interpolate(ind0, dind, u[ind0])


print((u1-mg1.u).sum())

plt.subplot(121)
plt.imshow(u1/(u1!=0), label='u')
plt.subplot(122)
plt.imshow(mg1.u, label='u')
plt.show()