import numpy as np
import matplotlib.pyplot as plt


from mgard import MGARD




#####################################################
# grid

# N = 2**4 + 1
# N = 2**8 + 1
N = 2**10 + 1

grid = np.linspace(0,1,N)
ind0 = np.arange(0,N,2)
dind = np.arange(1,N,2)


#####################################################
# data

u = np.sin(np.arange(N)/N*2*np.pi)
# u = (np.arange(N).astype(float))**3

# # random data
# u = np.zeros_like(grid)
# for i in range(1,100):
# 	u += np.random.randn()/i * np.sin(np.arange(N)/N*i*np.pi)
# u -= u.mean()


plt.figure(1)
plt.plot(u)
plt.gca().set_box_aspect(1)
plt.savefig('very_smooth_function.pdf', dpi=100, bbox_inches='tight')


########################
mg0 = MGARD(grid, u, order=0)
mg1 = MGARD(grid, u, order=1)
mg2 = MGARD(grid, u, order=2)


mg0.decompose_full()
mg1.decompose_full()
mg2.decompose_full()


########################

c0 = mg0.u_mg.copy()
c1 = mg1.u_mg.copy()
c2 = mg2.u_mg.copy()


plt.figure(2)
plt.semilogy(np.sort(np.abs(c0)), label='order 0')
plt.semilogy(np.sort(np.abs(c1)), label='order 1')
plt.semilogy(np.sort(np.abs(c2)), label='order 2')
plt.title('Coefficients')
plt.legend()
plt.gca().set_box_aspect(1)
plt.savefig('very_smooth_coef.pdf', dpi=100, bbox_inches='tight')


########################
plt.figure(3)

err_inf = {}
for mg in [mg0,mg1,mg2]:
	ind = np.argsort(np.abs(mg.u_mg))
	err_inf[mg] = []
	for i in range(len(ind)):
		mg.u_mg[ind[:i]] = 0
		u_mg = mg.u_mg.copy()
		mg.recompose_full()
		err_inf[mg].append(np.linalg.norm(u-mg.u_mg,ord=np.inf) / np.linalg.norm(u,ord=np.inf))
		mg.u_mg = u_mg

plt.semilogy(err_inf[mg0][-1::-1], label='order 0')
plt.semilogy(err_inf[mg1][-1::-1], label='order 1')
plt.semilogy(err_inf[mg2][-1::-1], label='order 2')
plt.title('Truncation error')
plt.xlabel('Retained coefficients', fontsize='xx-large')
plt.ylabel(r'$\|u-\tilde{u}\|_{\infty}/\|u\|_{\infty}$', fontsize='xx-large')
plt.gca().set_box_aspect(1)
plt.savefig('very_smooth_err.pdf', dpi=100, bbox_inches='tight')



plt.show()

