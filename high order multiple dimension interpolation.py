import numpy as np
import sys
import matplotlib.pyplot as plt

np.set_printoptions(precision=18)

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
				for m in range(0,self.order[d]):
					for k in range(m+1,self.order[d]+1):
						h_dic["h{0}{1}".format(m,k)] =  grid_d[ind0_d[i+m]] - grid_d[ind0_d[i+k]]

				for j in range(0,self.order[d]):
					l_dic = {}
					for r in range(self.order[d]+1):
						h_vector = []
						numer_vector = []
						l_vector = np.zeros(self.order[d]+1)
						for s in range(self.order[d]+1):
							if s != r:
								ls = grid_d[dind_d[i+j]] - grid_d[ind0_d[i+s]]
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

					interp_ind = np.ix_(*(i_f+[[dind_d[i+j]]]+i_c))

					interp_dic = {}
					for p in range(0,self.order[d]+1):
						interp_dic["ind_{0}".format(p)] = np.ix_(*(i_f+[[ind0_d[i+p]]]+i_c))

					
					self.u[interp_ind] = 0
					for (key,l) in zip(interp_dic,l_dic):
						self.u[interp_ind] = self.u[interp_ind] + self.u[interp_dic[key]]*l_dic[l]


		return self.u,


N = 6*2**6 + 1

dim = 2

grid = [np.logspace(0,1,N) for _ in range(dim)]
ind  = [np.arange(0,N)     for _ in range(dim)]
ind0 = [np.arange(0,N,2)   for _ in range(dim)]
dind = [np.arange(1,N,2)   for _ in range(dim)]




'''u = np.zeros((len(grid[0]),len(grid[1])))
for d in range(1,dim):
	for i in range(0,len(grid[0])):
		for j in range(0,len(grid[0])):
			u[i,j] = 0.26*(i**2 + j**2)-0.48*i*j'''

'''u = np.square(np.arange(0,N)) + np.arange(0,N)
for d in range(1,dim):
	v = np.square(np.arange(0,N)) + np.arange(0,N)
	u = u[...,np.newaxis] * v.reshape([1]*(d-1)+[len(v)])

print(u)'''


u = np.cos(np.arange(N)/N*2*np.pi)
for d in range(1,dim):
	v = np.cos(np.arange(N)/N*2*np.pi)
	u = u[...,np.newaxis] * v.reshape([1]*(d-1)+[len(v)])

#print(u)


mg1 = MGARD(grid, u, order=(1,1))

u1 = np.zeros_like(u)
u1[np.ix_(*ind)] = u[np.ix_(*ind)]
res_mg1 = mg1.interpolate_nd(ind, ind0, dind)

#print(mg1.u[1::2])

mg_1 = np.absolute(u1[1::2]-mg1.u[1::2]).flatten()

print((np.absolute(u1-mg1.u)).sum())

#plt.subplot(121)
#plt.imshow(u1, label='u')
#plt.subplot(122)
#plt.imshow(mg1.u, label='u')
#plt.show()


mg2 = MGARD(grid, u, order=(2,2))
res_mg2 = mg2.interpolate_nd(ind, ind0, dind)
#print((np.absolute(u2-mg2.u)).sum())
print((np.absolute(u1-mg2.u)).sum())
mg_2 = np.absolute(u1[1::2]-mg2.u[1::2]).flatten()


mg3 = MGARD(grid, u, order=(3,3))
res_mg3 = mg3.interpolate_nd(ind, ind0, dind)
print((np.absolute(u1-mg3.u)).sum())
mg_3 = np.absolute(u1[1::2]-mg3.u[1::2]).flatten()


mg4 = MGARD(grid, u, order=(4,4))
res_mg4 = mg4.interpolate_nd(ind, ind0, dind)
print((np.absolute(u1-mg4.u)).sum())
mg_4 = np.absolute(u1[1::2]-mg4.u[1::2]).flatten()


'''plt.subplot(121)
plt.plot(grid, u1,  label='u')
plt.step(grid, u0, label='u0')
plt.plot(label='u1')
plt.plot(label='u2')
plt.plot(label='u3')
plt.plot(label='u4')
plt.gca().set_box_aspect(1)
plt.gca().title.set_text('Interpolants')
plt.legend()'''

'''mg_1 = np.absolute(mg1.u-u1)[1]
mg_2 = np.absolute(mg2.u-u1)[1]
mg_3 = np.absolute(mg3.u-u1)[1]
mg_4 = np.absolute(mg4.u-u1)[1]'''

#plt.subplot(122)
#plt.semilogy(np.arange(p), mg_1[0:p],label='u1')
#plt.semilogy(np.arange(p), mg_2[0:p],label='u2')
#plt.semilogy(np.arange(p), mg_4[0:p],label='u4')
#plt.semilogy(np.arange(p), mg_8[0:p],label='u8')
#plt.gca().set_box_aspect(1)
#plt.gca().title.set_text('Errors (Sine)')
#plt.legend()
#plt.show()

p = 150

plt.subplot(122)
plt.semilogy(np.arange(p), mg_1[0:p], label='u1')
plt.semilogy(np.arange(p), mg_2[0:p],label='u2')
plt.semilogy(np.arange(p), mg_3[0:p],label='u3')
plt.semilogy(np.arange(p), mg_4[0:p],label='u4')
plt.gca().set_box_aspect(1)
plt.gca().title.set_text('Errors (Cosine)')
plt.legend()
plt.show()

