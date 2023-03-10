import numpy as np

import matplotlib.pyplot as plt


def Gramm_matrix(grid, order=1):
	''' Gramm matrix for piecewise polynomial of given order on a given grid '''

	if order==1:
		dgrid = np.diff(grid)
		main_diag = np.concatenate((dgrid[0:1]/3,(dgrid[:-1]+dgrid[1:])/3,dgrid[-1:]/3))
		return np.diag(dgrid/6,-1) + np.diag(main_diag) + np.diag(dgrid/6,1)



class MGARD(object):
	def __init__(self, grid, u, order=1, interp='mid'):
		self.grid   = grid
		self.u      = u
		self.u_mg   = u.copy()
		self.order  = order
		self.interp = interp


	def interpolate(self, ind0, dind, u0):
		'''Interpolate values from coarse grid to surplus grid

		Inputs
		------
		  ind0:	indices of the coarse  nodes
		  dind:	indices of the surplus nodes
		  u0:	values at the coarse nodes
		'''
		dgrid = np.diff(grid[ind0])

		if self.order==1:
			# interpolation matrix
			P = np.zeros((len(dind),len(ind0)))
			# left dof
			P[:,:-1] = np.diag((grid[ind0][1:]-grid[dind])/dgrid)
			# right dof
			P += np.diag((grid[dind]-grid[ind0][:-1])/dgrid,1)[:-1,:]
			return P @ u0
		elif self.order==0:
			# left dof (cadlag)
			# P[:,:-1] = np.eye(len(dind))
			# P[:,1:] = np.eye(len(dind))

			if self.interp=='mid':
				return 0.5*(u0[:-1]+u0[1:])
			else:
				return u0[:-1]
			# return u0[1:]
			# alpha = 0.3
			# return alpha*u0[:-1] + (1-alpha)*u0[1:]


	def project(self, ind0, dind, ud):
		'''Project function on a surplus grid to coarse grid

		Inputs
		------
		  ind0:	indices of the coarse  nodes
		  dind:	indices of the surplus nodes
		  ud:	values at the surplus nodes
		'''
		f = np.zeros(len(ind0))
		if self.order==1:
			G = Gramm_matrix(self.grid[ind0], self.order)
			#
			d0 = np.diff(self.grid[ind0])
			d1 = self.grid[ind0[1:]] - self.grid[dind]
			d2 = self.grid[dind]     - self.grid[ind0[:-1]]
			# contribution from dof to the left
			al = (2*d0 - d1) / 6
			# contribution from dof to the right
			ar = (2*d0 - d2) / 6
			# forcing term
			f[0]    =                     ar[0]  * ud[0]
			f[1:-1] = al[:-1] * ud[:-1] + ar[1:] * ud[1:]
			f[-1]   = al[-1]  * ud[-1]
			return np.linalg.solve(G,f)
		elif self.order==0:
			# contribution from dof to the left
			al = (self.grid[ind0][1:]-self.grid[dind]) / 2
			# contribution from dof to the right
			ar = (self.grid[ind0][1:]-self.grid[dind]) / 2
			# forcing term
			f[0]    =                     ar[0]  * ud[0]
			f[1:-1] = al[:-1] * ud[:-1] + ar[1:] * ud[1:]
			f[-1]   = al[-1]  * ud[-1]
			#
			diag = np.zeros((len(ind0),))
			diag[0]    = self.grid[dind][0]  - self.grid[ind0][0]
			diag[1:-1] = self.grid[dind][1:] - self.grid[dind][:-1]
			diag[-1]   = self.grid[ind0][-1] - self.grid[dind][-1]
			# diag[0]    = ar[0]
			# diag[1:-1] = ar[1:] + al[:-1]
			# diag[-1]   = al[-1]
			# print(diag)
			# exit()
			# print(f/diag)
			return f / diag


	# def decompose(self, ind0, dind):
	# 	# detail coefficients
	# 	ud = self.u[dind] - self.interpolate(ind0, dind, self.u[ind0])

	# 	# approximation coefficients
	# 	u0 = self.u[ind0] + self.project(ind0, dind, ud)
	# 	return u0, ud


	# def recompose(self, u0, ud, ind0, dind):
	# 	u = np.zeros_like(self.u)
	# 	u[ind0] = u0 - self.project(ind0, dind, ud)
	# 	u[dind] = self.interpolate(ind0, dind, u[ind0]) + ud
	# 	return u


	def decompose(self, ind0, dind):
		'''Compute approximation and detail coefficients at the given level

		Inputs
		------
		  ind0:	indices of the coarse  nodes
		  dind:	indices of the surplus nodes
		'''
		# detail coefficients
		self.u_mg[dind] -= self.interpolate(ind0, dind, self.u_mg[ind0])

		# approximation coefficients
		self.u_mg[ind0] += self.project(ind0, dind, self.u_mg[dind])
		return self.u_mg[ind0], self.u_mg[dind]


	def recompose(self, ind0, dind):
		'''Recompose data from approximation and detail coefficients at the given level

		Inputs
		------
		  ind0:	indices of the coarse  nodes
		  dind:	indices of the surplus nodes
		'''
		# approximation coefficients
		self.u_mg[ind0] -= self.project(ind0, dind, self.u_mg[dind])

		# detail coefficients
		self.u_mg[dind] += self.interpolate(ind0, dind, self.u_mg[ind0])
		return self.u_mg


	def decompose_grid(self):
		'''Decompose original grid into sequence of coarse and surplus grids
		'''
		self.grids = []
		ind0 = np.arange(self.grid.size)
		while len(ind0)>2:
			dind = ind0[1::2]
			ind0 = ind0[0::2]
			self.grids.append([ind0,dind])
		return self.grids

	def decompose_full(self):
		self.decompose_grid()

		u0 = self.u
		for ind0,dind in self.grids:
			u0,ud = self.decompose(ind0,dind)
		return self.u_mg


	def recompose_full(self):
		for ind0,dind in self.grids[len(self.grids)::-1]:
			self.recompose(ind0,dind)
		return self.u_mg


	def full_basis(self):
		grid = self.grid
		b = []
		for i,(ind0,dind) in enumerate(self.decompose_grid()):
			# compute surplus basis
			sb = self.surplus_basis(ind0, dind)
			# for each surplus basis function
			for sbi in sb:
				bi = sbi
				# interpolate it back to the original grid
				for ind01,dind1 in self.decompose_grid()[:i][::-1]:
					buf = np.zeros((len(ind01)+len(dind1),))
					buf[0::2] = bi[:]
					buf[1::2] = self.interpolate(ind01, dind1, buf[0::2])
					bi = buf
				b.append(bi)
		for sbi in np.eye(len(ind0)):
			bi = sbi
			# interpolate it back to the original grid
			for ind01,dind1 in self.decompose_grid()[:i+1][::-1]:
				buf = np.zeros((len(ind01)+len(dind1),))
				buf[0::2] = bi[:]
				buf[1::2] = self.interpolate(ind01, dind1, buf[0::2])
				bi = buf
			b.append(bi)
		b = np.vstack(b)
		return np.fliplr(np.flipud(b))
	# def full_basis(self):
	# 	grid = self.grid
	# 	bd = []
	# 	for ind0,dind in self.decompose_grid():
	# 		sb = self.surplus_basis(ind0, dind)
	# 		Qb[i,1::2] = self.interpolate(ind0, dind, Qb[i,0::2])
	# 		bd.append((grid,sb))
	# 		grid = self.grid[ind0]
	# 	# bd.append((grid,b0i))
	# 	return bd
	# def full_basis(self):
	# 	grid = self.grid
	# 	bd = []
	# 	for ind00,_ in self.decompose_grid():
	# 		ind0 = np.arange(0,grid.size,2)
	# 		dind = np.arange(1,grid.size,2)
	# 		b0i, bdi = self.basis_functions(ind0, dind, mode='mgard')
	# 		bd.append((grid,bdi))
	# 		grid = self.grid[ind00]
	# 	bd.append((grid,b0i))
	# 	return bd
	# def full_basis(self):
	# 	grid = self.grid
	# 	bd = []
	# 	for ind00,_ in self.grids:
	# 		ind0 = np.arange(0,grid.size,2)
	# 		dind = np.arange(1,grid.size,2)
	# 		b0i, bdi = self.basis_functions(ind0, dind, mode='mgard')
	# 		bd.append((grid,bdi))
	# 		grid = self.grid[ind00]
	# 	bd.append((grid,b0i))

	# 	bdd = [bd[0][1]]
	# 	# dind = np.arange(self.grid.size)
	# 	# ind0,dind = self.grids[0]
	# 	for i,(gi,bi) in enumerate(bd[1:]):
	# 		# bii = bi
	# 		# ind0,dind = self.grids[i+1]
	# 		bii = np.zeros((bi.shape[0],self.grid.size))
	# 		# print(bii.shape, bi.shape, ind0.size, dind.size)
	# 		bii[:,::2**(i+1)] = bi
	# 		for j in range(i+1):
	# 			# print(i,j)
	# 			ind0,dind = self.grids[j]
	# 			# print(ind0.shape,dind.shape)
	# 			for k in range(bii.shape[0]):
	# 				bii[k,1::2] = self.interpolate(ind0, dind, bii[k,::2])
	# 		bdd.append(bii)
	# 	bdd = np.vstack(bdd)
	# 	# print(bdd[0].shape)
	# 	# print(bdd[1].shape)
	# 	# print(bdd.shape)
	# 	# exit()
	# 	return bdd


	def surplus_basis(self, ind0, dind):
		'''Evaluate surplus basis functions on the level grid
		'''
		# fine nodal basis values on surplus grid
		phi = np.eye(len(dind))

		# values of the fine nodal basis on surplus grid
		b = np.zeros((len(dind),len(ind0)+len(dind)))
		# b[:,1::2] = 1
		for bi,di in zip(b,np.arange(b.shape[1])[1::2]):
			bi[di] = 1
		# print(b)
		# exit()

		if self.order==1:
			# fine basis on surplus grid projected to coarse grid
			Qb = np.zeros((len(dind),len(ind0)+len(dind)))
			for i in range(len(dind)):
				# project fine basis function on coarse basis
				Qb[i,0::2] = self.project(ind0, dind, phi[:,i])
				# interpolate back to surplus grid
				Qb[i,1::2] = self.interpolate(ind0, dind, Qb[i,0::2])

			# surplus basis
			b -= Qb
		return b
	# def basis_functions(self, ind0, dind, mode='mgard'):
	# 	'''Compute surplus basis functions
	# 	'''
	# 	# fine nodal basis values on surplus grid
	# 	phi = np.eye(len(dind))

	# 	# values of the fine nodal basis on surplus grid
	# 	b = np.zeros((len(dind),len(ind0)+len(dind)))
	# 	for bi,di in zip(b,dind):
	# 		bi[di] = 1

	# 	if self.order==1:
	# 		# fine basis on surplus grid projected to coarse grid
	# 		Qb = np.zeros((len(dind),len(ind0)+len(dind)))
	# 		for i in range(len(dind)):
	# 			# project fine basis function on coarse basis
	# 			Qb[i,ind0] = self.project(ind0, dind, phi[:,i])
	# 			# interpolate back to surplus grid
	# 			Qb[i,dind] = self.interpolate(ind0, dind, Qb[i,ind0])

	# 		# surplus basis
	# 		b -= Qb
	# 	elif self.order==0:
	# 		# fine basis on surplus grid projected to coarse grid
	# 		Qb = np.zeros((len(dind),len(ind0)+len(dind)))
	# 		for i in range(len(dind)):
	# 			# project fine basis function on coarse basis
	# 			Qb[i,ind0] = self.project(ind0, dind, phi[:,i])
	# 			# interpolate back to surplus grid
	# 			Qb[i,dind] = self.interpolate(ind0, dind, Qb[i,ind0])

	# 		# fine basis on surplus grid
	# 		b = np.zeros_like(Qb)
	# 		for bi,di in zip(b,dind):
	# 			bi[di] = 1

	# 		# surplus basis
	# 		if mode=='mgard':
	# 			b -= Qb
	# 		# b = Qb

	# 		# print(b[0])

	# 	return [np.eye(len(ind0)), b]






def basis_function(basis, level, index):
	if level==0:
		lind = 0
		uind = 2
	elif level>0:
		lind = 2 + 2**(level-1) - 1
		uind = 2 + 2**(level)   - 1
	level_basis = basis[lind:uind,:]
	return level_basis[index,:]



if __name__ == '__main__':
	fig_no = 1


	# N = 2**3 + 1
	# N = 2**5 + 1
	N = 2**10 + 1


	grid = np.linspace(0,1,N)
	# ind0 = slice(0,N,2)
	# dind = slice(1,N,2)
	ind0 = np.arange(0,N,2)
	dind = np.arange(1,N,2)

	edges  = [grid[0]] + list(0.5*(grid[:-1]+grid[1:])) + [grid[-1]]
	edges0 = [grid[ind0][0]] + list(0.5*(grid[ind0][:-1]+grid[ind0][1:])) + [grid[ind0][-1]]
	# edgesd = 0.5*(grid[ind0][:-1]+grid[ind0][1:])) + [grid[ind0][-1]]

	#####################################################
	# data

	u = np.sin(np.arange(N)/N*20*np.pi)
	# u = (np.arange(N).astype(float))**3

	# random data
	u = np.zeros_like(grid)
	for i in range(1,50):
		u += np.random.randn()/i * np.sin(np.arange(N)/N*i*np.pi)
	u -= u.mean()


	#####################################################

	# # du = interpolate(u[ind0], grid, ind0, dind)
	# # print(du)
	# plt.plot(grid, u, '-', linewidth=3)
	# # plt.plot(grid[ind0], u[ind0], '-')
	# # plt.plot(grid[dind], du, 'x')
	# plt.savefig('tmp.png', dpi=100, bbox='tight')
	# plt.show()
	# exit()


	########################
	mg0  = MGARD(grid, u, order=0)
	mg0l = MGARD(grid, u, order=0, interp='left')
	mg1  = MGARD(grid, u, order=1)


	mg0.decompose_full()
	mg0l.decompose_full()
	mg1.decompose_full()


	########################


	# ind = np.argsort(np.abs(mg0.u_mg))
	# u_mg = mg0.u_mg.copy()
	# error0_inf = []
	# indicator0_inf = []
	# dropped = []
	# C2 = 1+3
	# for i in ind:
	# 	dropped.append(mg0.u_mg[i])
	# 	mg0.u_mg[i] = 0
	# 	indicator0_inf.append( C2 * len(mg0.grids) * np.linalg.norm(dropped,ord=np.inf) / np.linalg.norm(u,ord=np.inf))
	# 	error0_inf.append(np.linalg.norm(u-mg0.recompose_full(),ord=np.inf) / np.linalg.norm(u,ord=np.inf))
	# 	mg0.u_mg = u_mg.copy()

	# ind = np.argsort(np.abs(mg0l.u_mg))
	# u_mg = mg0l.u_mg.copy()
	# error0l_inf = []
	# indicator0l_inf = []
	# dropped = []
	# C2 = 1+3
	# for i in ind:
	# 	dropped.append(mg0l.u_mg[i])
	# 	mg0l.u_mg[i] = 0
	# 	indicator0l_inf.append( C2 * len(mg0l.grids) * np.linalg.norm(dropped,ord=np.inf) / np.linalg.norm(u,ord=np.inf))
	# 	error0l_inf.append(np.linalg.norm(u-mg0l.recompose_full(),ord=np.inf) / np.linalg.norm(u,ord=np.inf))
	# 	mg0l.u_mg = u_mg.copy()

	# ind = np.argsort(np.abs(mg1.u_mg))
	# u_mg = mg1.u_mg.copy()
	# error1_inf = []
	# indicator1_inf = []
	# dropped = []
	# C2 = 1+3
	# for i in ind:
	# 	dropped.append(mg1.u_mg[i])
	# 	mg1.u_mg[i] = 0
	# 	indicator1_inf.append( C2 * len(mg1.grids) * np.linalg.norm(dropped,ord=np.inf) / np.linalg.norm(u,ord=np.inf))
	# 	error1_inf.append(np.linalg.norm(u-mg1.recompose_full(),ord=np.inf) / np.linalg.norm(u,ord=np.inf))
	# 	mg1.u_mg = u_mg.copy()

	# # plt.semilogy(np.arange(len(error0_inf))[::20], error0_inf[-1::-20], 'x',  label='order 0, mid. point')

	# plt.semilogy(error1_inf[-1::-1],  label='order 1 (current)')
	# plt.semilogy(indicator1_inf[-1::-1], '--',  label='order 1 (current)')
	# plt.semilogy(indicator0_inf[-1::-1], '--',  label='order 1 (current)')
	# plt.semilogy(indicator0l_inf[-1::-1], '--',  label='order 1 (current)')
	# plt.semilogy(error0_inf[-1::-1],  label='order 0, mid. point')
	# plt.semilogy(error0l_inf[-1::-1], label='order 0, left point')
	# plt.legend(framealpha=1.0, fontsize='x-large')
	# plt.xlabel('Retained degrees of freedom', fontsize='x-large')
	# plt.ylabel(r'$\|u-\tilde{u}\|_{\infty}/\|u\|_{\infty}$', fontsize='xx-large')
	# plt.savefig('tmp.png', dpi=100, bbox='tight')
	# plt.show()

	# exit()



	########################
	# plot coeffiecient decay

	plt.figure(fig_no); fig_no += 1

	plt.semilogy(np.sort(np.abs(mg0.u_mg))[::-1],  label='Order 0')
	plt.semilogy(np.sort(np.abs(mg0l.u_mg))[::-1], label='Order 0l')
	plt.semilogy(np.sort(np.abs(mg1.u_mg))[::-1],  label='Order 1')
	plt.title('Sorted coefficients')
	plt.legend()
	plt.gca().set_box_aspect(1)

	# exit()


	# ########################

	# 	# order 0
	# u00, ud0 = mg0.decompose(ind0, dind)
	# u10 = mg0.recompose(u00, ud0, ind0, dind)

	# # order 1
	# u01, ud1 = mg1.decompose(ind0, dind)
	# u11 = mg1.recompose(u01, ud1, ind0, dind)


	# plt.figure(fig_no); fig_no += 1

	# plt.subplot(2,3,1)
	# # plt.stairs(u,   edges=edges,  baseline=None)
	# # plt.stairs(u00, edges=edges0, baseline=None)
	# # plt.step(grid, u, where='mid')
	# plt.plot(grid, u)
	# plt.step(grid[ind0], u00, where='mid')
	# # plt.stairs(ud0, edges=edgesd, baseline=None)
	# plt.plot(grid[dind], ud0, '-x')
	# # plt.legend(['u','u0','ud'])
	# plt.gca().set_box_aspect(1)

	# plt.subplot(2,3,2)
	# plt.plot(grid, u-u10)
	# plt.gca().set_box_aspect(1)

	# plt.subplot(2,3,4)
	# plt.plot(grid, u)
	# plt.plot(grid[ind0], u01, '-')
	# plt.plot(grid[dind], ud1, '-x')
	# # # plt.legend(['u','u0','ud'])
	# plt.gca().set_box_aspect(1)

	# plt.subplot(2,3,5)
	# plt.plot(grid, u-u11)
	# plt.gca().set_box_aspect(1)

	# plt.subplot(2,3,6)
	# plt.semilogy(np.sort(np.abs(np.hstack((u00,ud0)))))
	# plt.semilogy(np.sort(np.abs(np.hstack((u01,ud1)))))
	# plt.gca().set_box_aspect(1)

	# plt.show()

	# # print(np.abs(np.hstack((u00,ud0))).size)
	# # print(np.abs(np.hstack((u01,ud1))).size)


	# # plt.figure(3)

	# # print("Hello")
	# exit()


	########################################################
	# nodal basis

	nodal_basis_0 = np.eye(grid.size)
	for i in range(nodal_basis_0.shape[0]):
		for j in range(nodal_basis_0.shape[1]):
			if np.abs(i-j)>1: nodal_basis_0[i,j] = np.nan

	nodal_basis_1 = np.eye(grid.size)
	for i in range(nodal_basis_1.shape[0]):
		for j in range(nodal_basis_1.shape[1]):
			if np.abs(i-j)>1: nodal_basis_1[i,j] = np.nan

	# plt.figure(fig_no); fig_no += 1
	# for nb in nodal_basis_1:
	# 	plt.plot(grid, nb)
	# 	# plt.step(grid, nb, where='mid')
	# plt.show()
	# exit()


	########################################################
	# mgard basis

	# plt.figure(fig_no); fig_no += 1

	# mg_basis_0 = mg0.full_basis()
	mg_basis_1 = mg1.full_basis()

	# np.save('basis', mg_basis_1)
	# exit()

	# basis = np.zeros((grid.size,grid.size))

	# # print(mg_basis_0.shape)
	# i = 0
	# l = 0
	# for bii in mg_basis_0:
	# 	for bi in bii[1]:
	# 		basis[i,::2**l] = bi
	# 		# for j in range():
	# 		# 	basis[i,1::2**l] = basis[i,::2**l]
	# 		# print(bi.shape)
	# 		i += 1
	# 	l += 1
	# print(basis)
	# plt.step(grid, basis[-1,:], where='mid')
	# plt.show()
	# exit()


	# print(mg_basis_0[-1][0])
	# # print(mg_basis_1[-2][1])
	# exit()

	# L = 4

	# plt.figure(fig_no, figsize=(3*L,3)); fig_no += 1
	# ax1 = plt.subplot(1,L+2,1)
	# for nb in nodal_basis_1[-14:-13]:
	# 	plt.plot(grid, nb, color='g')
	# plt.gca().set_box_aspect(1)
	# plt.gca().spines['bottom'].set_position('zero')
	# plt.gca().spines['top'].set_color('none')
	# plt.gca().spines['right'].set_color('none')
	# plt.gca().set_xticks(grid)
	# plt.gca().set_xticklabels([]*len(grid))
	# plt.gca().set_yticklabels([])
	#
	# for l in range(L):
	# 	plt.subplot(1,L+2,l+2, sharey=ax1)
	# 	mid_i = mg_basis_1[-l-1][1].shape[0]//2
	# 	for bdi in mg_basis_1[0][1][-7:-6]:
	# 		plt.plot(mg_basis_1[0][0],bdi)
	# 	for bdi in mg_basis_1[-l-1][1][mid_i:mid_i+1]:
	# 		plt.plot(mg_basis_1[-l-1][0],bdi)
	# 	plt.gca().set_box_aspect(1)
	# 	plt.gca().spines['bottom'].set_position('zero')
	# 	plt.gca().spines['top'].set_color('none')
	# 	plt.gca().spines['right'].set_color('none')
	# 	plt.gca().set_xticks(grid)
	# 	plt.gca().set_xticklabels([]*len(grid))
	# 	plt.gca().set_yticklabels([])
	# plt.subplot(1,L+2,L+2, sharey=ax1)
	# mid_i = mg_basis_1[0][1].shape[0]//2
	# for bdi in mg_basis_1[0][1][-7:-6]:
	# 	plt.plot(mg_basis_1[0][0],bdi)
	# for bdi in mg_basis_1[0][1][mid_i:mid_i+1]:
	# 	plt.plot(mg_basis_1[0][0],bdi)
	# plt.gca().set_box_aspect(1)
	# plt.gca().spines['bottom'].set_position('zero')
	# plt.gca().spines['top'].set_color('none')
	# plt.gca().spines['right'].set_color('none')
	# plt.gca().set_xticks(grid)
	# plt.gca().set_xticklabels([]*len(grid))


	plt.figure(fig_no); fig_no += 1
	plt.plot(basis_function(mg_basis_1, 0, 0))
	plt.plot(basis_function(mg_basis_1, 0, 1))
	plt.gca().set_box_aspect(1)

	plt.figure(fig_no); fig_no += 1
	plt.plot(basis_function(mg_basis_1, 0, 0))
	plt.plot(basis_function(mg_basis_1, 4, 0))
	plt.plot(basis_function(mg_basis_1, 4, -1))
	plt.gca().set_box_aspect(1)

	plt.show()
	exit()


	plt.subplot(1,L+2,l+2, sharey=ax1)
	for bdi in mg_basis_1[0][1][-7:-6]:
		plt.plot(basis_function(basis, level, index))
	for bdi in mg_basis_1[-l-1][1][mid_i:mid_i+1]:
		plt.plot(mg_basis_1[-l-1][0],bdi)
	plt.gca().set_box_aspect(1)
	plt.gca().spines['bottom'].set_position('zero')
	plt.gca().spines['top'].set_color('none')
	plt.gca().spines['right'].set_color('none')
	plt.gca().set_xticks(grid)
	plt.gca().set_xticklabels([]*len(grid))
	plt.gca().set_yticklabels([])

	plt.subplot(1,L+2,L+2, sharey=ax1)
	mid_i = mg_basis_1[0][1].shape[0]//2
	for bdi in mg_basis_1[0][1][-7:-6]:
		plt.plot(mg_basis_1[0][0],bdi)
	for bdi in mg_basis_1[0][1][mid_i:mid_i+1]:
		plt.plot(mg_basis_1[0][0],bdi)
	plt.gca().set_box_aspect(1)
	plt.gca().spines['bottom'].set_position('zero')
	plt.gca().spines['top'].set_color('none')
	plt.gca().spines['right'].set_color('none')
	plt.gca().set_xticks(grid)
	plt.gca().set_xticklabels([]*len(grid))

	plt.show()

	exit()

	plt.figure(figsize=(3*L,3))
	ax1 = plt.subplot(1,L+2,1)
	for nb in nodal_basis_0[-14:-13]:
		plt.step(grid, nb, color='g',where='mid')
	plt.gca().set_box_aspect(1)
	plt.gca().spines['bottom'].set_position('zero')
	plt.gca().spines['top'].set_color('none')
	plt.gca().spines['right'].set_color('none')
	plt.gca().set_xticks(grid)
	plt.gca().set_xticklabels([]*len(grid))
	plt.gca().set_yticklabels([])

	for l in range(L):
		plt.subplot(1,L+2,l+2, sharey=ax1)
		mid_i = mg_basis_0[-l-1][1].shape[0]//2
		for bdi in mg_basis_0[0][1][-7:-6]:
			bdi /= (bdi!=0)
			plt.step(mg_basis_1[0][0],bdi,where='mid')
		for bdi in mg_basis_0[-l-1][1][mid_i:mid_i+1]:
			bdi /= (bdi!=0)
			plt.step(mg_basis_0[-l-1][0],bdi,where='mid')
		plt.gca().set_box_aspect(1)
		plt.gca().spines['bottom'].set_position('zero')
		plt.gca().spines['top'].set_color('none')
		plt.gca().spines['right'].set_color('none')
		plt.gca().set_xticks(grid)
		plt.gca().set_xticklabels([]*len(grid))
		plt.gca().set_yticklabels([])
	plt.subplot(1,L+2,L+2, sharey=ax1)
	mid_i = mg_basis_0[0][1].shape[0]//2
	for bdi in mg_basis_0[0][1][-7:-6]:
		bdi /= (bdi!=0)
		plt.step(mg_basis_0[0][0],bdi,where='mid')
	for bdi in mg_basis_0[0][1][mid_i:mid_i+1]:
		bdi /= (bdi!=0)
		plt.step(mg_basis_0[0][0],bdi,where='mid')
	plt.gca().set_box_aspect(1)
	plt.gca().spines['bottom'].set_position('zero')
	plt.gca().spines['top'].set_color('none')
	plt.gca().spines['right'].set_color('none')
	plt.gca().set_xticks(grid)
	plt.gca().set_xticklabels([]*len(grid))


	plt.show()

	exit()


	########################################################

	plt.subplot(2,1,1)
	b0, bd = mg0.basis_functions(ind0, dind, mode='mgard')
	b0 /= (b0!=0)
	bd /= (bd!=0)

	# print(Qb[0])
	# plt.step(grid,Qb[0],where='mid')
	for b0i in b0:
		plt.step(grid[ind0],b0i,where='mid')
	for bdi in bd:
		plt.step(grid,bdi,where='mid')
	# plt.step(grid[ind0],b0[1],where='mid')
	# plt.step(grid,bd[1],where='mid')
	# plt.plot(grid,bd[1])
	# plt.step(grid,Qb[2],where='mid')
	# # plt.plot(grid,Qb[0])
	# plt.step(grid,Qb[dind.size//2],where='mid')
	# # plt.plot(grid,Qb[dind.size//2])
	# plt.step(grid,Qb[-1],where='mid')
	# # plt.plot(grid,Qb[-1])
	plt.xlim([grid[0],grid[-1]])

	plt.subplot(2,1,2)
	b0, bd = mg1.basis_functions(ind0, dind, mode='mgard')
	for i in range(b0.shape[0]):
		for j in range(b0.shape[1]):
			if np.abs(i-j)>1: b0[i,j] = np.nan
	for b0i in b0:
		plt.plot(grid[ind0],b0i)
	for bdi in bd:
		plt.plot(grid,bdi)
	# # plt.plot(grid,Qb[0])
	# plt.plot(grid,Qb[1])
	# # plt.plot(grid,Qb[dind.size//2-1])
	# # plt.plot(grid,Qb[-1])


	plt.show()






