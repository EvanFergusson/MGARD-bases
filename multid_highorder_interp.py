import numpy as np
import sys
import matplotlib.pyplot as plt

#np.set_printoptions(threshold=sys.maxsize)

class MGARD(object):
	def __init__(self, grid, u, order=1, interp='left'):
		self.grid   = grid
		self.u      = u
		self.u_mg   = u.copy()
		self.order  = order
		self.interp = interp

		self.ndim = u.ndim

		if interp!='left':
			raise ValueError("Avoid mid iterpolation at the moment")


	def interpolate_nd(self, ind, ind0, dind):
		'''Interpolate values from coarse grid to surplus grid in-place

		Inputs
		------
		  ind:	indices of the fine    nodes in each dimension
		  ind0:	indices of the coarse  nodes in each dimension
		  dind:	indices of the surplus nodes in each dimension
		'''

		# loop through dimensions
		for d in range(self.ndim):
			# coarse and surplus indices along the given dimension
			ind0_d = ind0[d]
			dind_d = dind[d]
			ind_d = ind[d]

			# 1d grid along the given dimension
			grid_d = self.grid[d]


			# loop through the 1d-elements along the given dimension
			for i in range(0,len(dind_d),self.order[d]):
				h_dic = {}
				for j in range(0,self.order[d]):
					for k in range(j+1,self.order[d]+1):
						h_dic["h{0}{1}".format(j,k)] =  grid_d[ind0_d[i+j]] - grid_d[ind0_d[i+k]]

					#print('h',h_dic)

					l_dic = {}
					for r in range(self.order[d]+1):
						h_vector = []
						numer_vector = []
						l_vector = np.zeros(self.order[d]+1)
						print(l_vector)
						for s in range(self.order[d]+1):
							if s != r:
								ls = grid_d[dind_d[i]] - grid_d[ind0_d[i+s]]
								numer_vector = np.append(numer_vector, ls)
						for h in h_dic:
							if int(h[1])==r or int(h[2])==r:
								h_vector = np.append(h_vector,h_dic[h])
						if r%2 == 1:
							l_dic["l{0}".format(r)] = - np.prod(numer_vector)/np.prod(h_vector)
						elif r%2 == 0:
							l_dic["l{0}".format(r)] = np.prod(numer_vector)/np.prod(h_vector)
					
					#print(l_dic)

					i_f = [i0 for i0 in ind[:d]]
					i_c = [i0 for i0 in ind0[d+1:]]

					interp_ind = np.ix_(*(i_f+[[dind_d[i+0]]]+i_c))

					interp_dic = {}
					for p in range(0,self.order[d]+1):
						interp_dic["ind_{0}".format(p)] = np.ix_(*(i_f+[[ind0_d[i+p]]]+i_c))


					print(l_vector)
					
					self.u[interp_ind] = 0
					p = 0
					for (key,l) in zip(interp_dic,l_dic):
						self.u[interp_ind] = self.u[interp_ind] + self.u[interp_dic[key]]*l_dic[l]
						p = p + 1
					

		return self.u,


N = 2**5 + 1

dim = 2

grid = [np.linspace(0,1,N) for _ in range(dim)]
ind  = [np.arange(0,N)     for _ in range(dim)]
ind0 = [np.arange(0,N,2)   for _ in range(dim)]
dind = [np.arange(1,N,2)   for _ in range(dim)]


u = np.cos(np.arange(N)/N*2*np.pi)
for d in range(1,dim):
	v = np.cos(np.arange(N)/N*2*np.pi)
	u = u[...,np.newaxis] * v.reshape([1]*(d-1)+[len(v)])


mg1 = MGARD(grid, u, order=(1,1))

u1 = np.zeros_like(u)
u1[np.ix_(*ind)] = u[np.ix_(*ind)]
res = mg1.interpolate_nd(ind, ind0, dind)

print(np.absolute(u1-mg1.u))


print((np.absolute(u1-mg1.u)).sum())

#plt.subplot(121)
#plt.imshow(u1/(u1!=0), label='u')
plt.subplot(122)
plt.imshow(mg1.u, label='u')
plt.show()
