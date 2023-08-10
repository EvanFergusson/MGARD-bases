import numpy as np
import matplotlib.pyplot as plt


from mgard import MGARD




#####################################################
# grid

L = 9

# N = 6 * 2**1 + 1
N = 2**L + 1

dim = 2

grid = [np.linspace(0,1,N) for _ in range(dim)]
ind  = [np.arange(0,N)     for _ in range(dim)]
ind0 = [np.arange(0,N,2)   for _ in range(dim)]
dind = [np.arange(1,N,2)   for _ in range(dim)]


order = 1


#####################################################
# data


u = np.cos(np.arange(N)/N*2*np.pi)



# random data
nr = 1
u = np.zeros_like(u)
for i in range(1,nr+1):
	u += np.random.randn()/i * np.sin(np.arange(N)/N*i*2*np.pi)
u -= u.mean()

for d in range(1,dim):
	# v = np.cos(np.arange(N)/N*2*np.pi)
	v = np.zeros_like(u)
	for i in range(1,nr+1):
		v += np.random.randn()/i * np.sin(np.arange(N)/N*i*2*np.pi)
	v -= v.mean()

	u = u[...,np.newaxis] * v.reshape([1]*(d-1)+[len(v)])

u_orig = u.copy()

# # random data
# u = np.zeros_like(grid)
# for i in range(1,50):
# 	u += np.random.randn()/i * np.sin(np.arange(N)/N*i*np.pi)
# u -= u.mean()


########################
# # mg0 = MGARD(grid, u, order=0)
# mg1 = MGARD(grid, u, order=[order]*dim)
# # mg2 = MGARD(grid, u, order=2)
# # mg3 = MGARD(grid, u, order=3)
# # mg4 = MGARD(grid, u, order=4)


# # for ei, si in mg1.split_element(ind):
# # 	print(ei, si)

# # # print( list(mg1.coarse_elements_iterator([np.arange(5),np.arange(5)], order=(2,1))) )
# # mg1.assemble_rhs(ind0,ind)
# # exit()


# # u0 = u.copy()
# # u0[dind] = mg0.interpolate(ind0, dind, u[ind0])

# indc = np.ix_(*ind0)

# u1 = np.zeros_like(u)
# u1[indc] = u[indc]
# mg1.interpolate_nd(ind, ind0, dind)

# # u2 = np.zeros_like(u)
# # u2[np.ix_(*ind0)] = u[np.ix_(*ind0)]
# # mg2.interpolate_nd(ind, ind0, dind)

# # mg1.project_nd(ind, ind0, dind)
# # print("Hello")
# # exit()

# # u2 = u.copy()
# # u2[dind] = mg2.interpolate(ind0, dind, u[ind0])

# # u3 = u.copy()
# # u3[dind] = mg3.interpolate(ind0, dind, u[ind0])

# # u4 = u.copy()
# # u4[dind] = mg4.interpolate(ind0, dind, u[ind0])


# print((np.abs(u_orig[indc]-mg1.u[indc])).sum())
# print((np.abs(u_orig-mg1.u)).sum())


# # plt.subplot(121)
# # plt.imshow(u1/(u1!=0), label='u')
# # plt.subplot(122)
# # plt.imshow(mg1.u, label='u')
# # plt.show()


# ##############


# mg1 = MGARD(grid, u, order=[order]*dim)

# u = mg1.project_nd(ind, ind0, dind)
# print((np.abs(u_orig[indc]-u)).sum())
# # print((np.abs(u_orig-mg1.u)).sum())

# # plt.subplot(121)
# # plt.imshow(u1/(u1!=0), label='u')
# # plt.subplot(122)
# # plt.imshow(mg1.u, label='u')
# # plt.show()


##############


from skimage.io import imread
from skimage.transform import resize
from skimage.filters import gaussian as skgaus

# u_orig = imread('chair.png', as_gray=True)
u_orig = imread('Convection.png', as_gray=True)
u_orig = u_orig[:,u_orig.shape[0]:2*u_orig.shape[0]]
# plt.imshow(u_orig)
# plt.show()
# exit()
print(u_orig.shape)
u_orig = resize(u_orig, (N,N))
# u_orig = resize(u_orig, (N//10,N//10))
# u_orig = resize(u_orig, (N,N))
u_orig = skgaus(u_orig, sigma=5)
print(N)


plt.figure()
for o in range(3):
	mg1 = MGARD(grid, u_orig, order=[order]*dim, order2=[o]*dim)
	# mg1.decompose(ind, ind0, dind)
	# mg1.recompose(ind, ind0, dind)
	# print(mg1.u_mg)
	# print((mg1.u_mg-u_orig).sum())
	# plt.imshow(u_orig-mg1.u_mg, label='u',interpolation='none')
	# plt.show()
	# exit()
	mg1.decompose_full()

	img = np.zeros_like(u_orig)
	for l in range(len(mg1.grids[0])):
		indf = [mg1.grids[d][l][0] for d in range(mg1.ndim)]
		indc = [mg1.grids[d][l][1] for d in range(mg1.ndim)]
		dind = [mg1.grids[d][l][2] for d in range(mg1.ndim)]
		img[:len(indc[0]),:len(indc[1])] = mg1.u_mg[np.ix_(indc[0],indc[1])]
		img[:len(indc[0]),len(indc[1]):len(indc[1])+len(dind[1])] = mg1.u_mg[np.ix_(indc[0],dind[1])]
		img[len(indc[0]):len(indc[0])+len(dind[0]),:len(indc[1])] = mg1.u_mg[np.ix_(dind[0],indc[1])]
		img[len(indc[0]):len(indc[0])+len(dind[0]),len(indc[1]):len(indc[1])+len(dind[1])] = mg1.u_mg[np.ix_(dind[0],dind[1])]
	img = np.abs(img)

	plt.semilogy(np.sort(img.ravel())[::-1], label=f'order {o}')
	plt.ylim([1.e-7,10])
plt.legend()


from matplotlib.colors import LogNorm
for o in range(3):
	mg1 = MGARD(grid, u_orig, order=[order]*dim, order2=[o]*dim)
	# mg1.decompose(ind, ind0, dind)
	mg1.decompose_full()


	img = np.zeros_like(u_orig)
	for l in range(len(mg1.grids[0])):
		indf = [mg1.grids[d][l][0] for d in range(mg1.ndim)]
		indc = [mg1.grids[d][l][1] for d in range(mg1.ndim)]
		dind = [mg1.grids[d][l][2] for d in range(mg1.ndim)]
		img[:len(indc[0]),:len(indc[1])] = mg1.u_mg[np.ix_(indc[0],indc[1])]
		img[:len(indc[0]),len(indc[1]):len(indc[1])+len(dind[1])] = mg1.u_mg[np.ix_(indc[0],dind[1])]
		img[len(indc[0]):len(indc[0])+len(dind[0]),:len(indc[1])] = mg1.u_mg[np.ix_(dind[0],indc[1])]
		img[len(indc[0]):len(indc[0])+len(dind[0]),len(indc[1]):len(indc[1])+len(dind[1])] = mg1.u_mg[np.ix_(dind[0],dind[1])]
	img = np.abs(img)
	# img[img>1] = 1.0
	# img /= img.max()

	mg1.recompose_full()

	plt.figure()
	# plt.subplot(121)
	plt.imshow(mg1.u_mg, label='u')
	# plt.subplot(122)
	# plt.imshow(mg1.u_mg[indc], label='u')
	# plt.title(f'order {o}')
	# plt.imshow(img, label='u',interpolation='none',norm=LogNorm(vmin=1.e-5, vmax=10))
	# plt.imshow(img, label='u',interpolation='none')
	plt.savefig(f'u.png', dpi=100, bbox_inches='tight')

# plt.show()
# exit()

# plt.figure()
# plt.imshow(mg1.u_mg, label='u',interpolation='none')



##########
# error

plt.figure()
for o in range(3):
	mg1 = MGARD(grid, u_orig, order=[order]*dim, order2=[o]*dim)
	mg1.decompose_full()
	umg = mg1.u_mg.copy().ravel()
	ind = np.argsort(np.abs(umg))

	nord = np.inf

	mg1.recompose_full()
	erry = [np.linalg.norm(u_orig-mg1.u_mg,ord=nord) / np.linalg.norm(u_orig,ord=nord)]
	errx = [len(ind)]
	# errx = [1]

	for i in range(0,len(ind),100):
	# for i in range(0,len(ind),100):
		for j in ind[:i+1]:
			umg[j] = 0
		mg1.u_mg.ravel()[...] = umg[...]
		mg1.recompose_full()
		errx.append(len(ind)-i-1)
		# errx.append(len(ind)/(len(ind)-i-1))
		erry.append(np.linalg.norm(u_orig-mg1.u_mg,ord=nord) / np.linalg.norm(u_orig,ord=nord))

	plt.semilogx(erry,errx, label=f'order {o}')
plt.legend()
plt.xlim([1.e-8,1])
# plt.ylabel('Retained coefficients', fontsize='large')
plt.ylabel('Compression ratio', fontsize='x-large')
plt.xlabel(r'$\|u-\tilde{u}\|_{\infty}/\|u\|_{\infty}$', fontsize='x-large')
plt.savefig(f'err_sm.png', dpi=100, bbox_inches='tight')



# ##########
# # basis

# mg1 = MGARD(grid, u_orig, order=[0]*dim)
# mg1.decompose_full()
# mg1.u_mg[...] = 0

# B = np.zeros((len(mg1.u_mg.ravel()),len(mg1.u_mg.ravel())))
# for i in range(len(mg1.u_mg.ravel())):
# 	mg1.u_mg.ravel()[i] = 1
# 	mg1.recompose_full()
# 	B[:,i] = mg1.u_mg.ravel()[...]
# 	# break
# 	# mg1.u_mg.ravel()[i] = 0
# print(B)
# plt.figure()
# plt.imshow(mg1.u_mg, label='u',interpolation='none')
# # plt.plot(B[:,0])
# plt.show()
# exit()


##########
# basis

plt.figure()

order = 2
ndof = (N-1)//4 - (N-1)//32 #+ (N-1)//64
# plt.figure()

mg1 = MGARD(grid, u_orig, order=[order]*dim)
mg1.decompose_full()

mg1.u_mg[...] = 0
mg1.u_mg[ndof,ndof] = 1
mg1.recompose_full()
mg1.u_mg[mg1.u_mg==0] = np.nan

plt.figure()
plt.imshow(mg1.u_mg, interpolation='none')
plt.savefig(f'2d_basis_{order}.png', dpi=100, bbox_inches='tight')

plt.plot(mg1.u_mg[ndof,:], label='order 0')
plt.xlim([0,N-1])


order = 1
ndof = (N-1)//2 - (N-1)//32 #+ (N-1)//64
# plt.figure()

mg1 = MGARD(grid, u_orig, order=[order]*dim)
mg1.decompose_full()

mg1.u_mg[...] = 0
mg1.u_mg[ndof,ndof] = 1
mg1.recompose_full()
mg1.u_mg[mg1.u_mg==0] = np.nan

# plt.figure()
# plt.imshow(mg1.u_mg, label='u',interpolation='none')

plt.plot(mg1.u_mg[ndof,:], label='order 1')
plt.xlim([0,N-1])



order = 2
ndof = 3*(N-1)//4 + (N-1)//32 #+ (N-1)//64
# plt.figure()

mg1 = MGARD(grid, u_orig, order=[order]*dim)
mg1.decompose_full()

mg1.u_mg[...] = 0
mg1.u_mg[ndof,ndof] = 1
mg1.recompose_full()
mg1.u_mg[mg1.u_mg==0] = np.nan

# plt.figure()
# plt.imshow(mg1.u_mg, label='u',interpolation='none')

plt.plot(mg1.u_mg[ndof,:], label='order 2')
plt.xlim([0,N-1])

plt.legend(fontsize='large')
plt.savefig('basis.png', dpi=100, bbox_inches='tight')


plt.show()