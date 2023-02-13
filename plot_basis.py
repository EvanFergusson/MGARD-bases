import numpy as np
import matplotlib.pyplot as plt



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

	# load precomputed basis
	basis = np.load('basis.npy')


	plt.figure()
	plt.plot(basis_function(basis, 0, 0))
	plt.plot(basis_function(basis, 6, 16))
	plt.gca().set_box_aspect(1)


	plt.figure()
	plt.plot(basis_function(basis, 1, 0))
	plt.plot(basis_function(basis, 6, 16))
	plt.gca().set_box_aspect(1)


	plt.figure()
	plt.plot(basis_function(basis, 2, 0))
	plt.plot(basis_function(basis, 6, 16))
	plt.gca().set_box_aspect(1)


	plt.figure()
	plt.plot(basis_function(basis, 3, 1))
	plt.plot(basis_function(basis, 6, 16))
	plt.gca().set_box_aspect(1)


	plt.figure()
	plt.plot(basis_function(basis, 6, 14))
	plt.plot(basis_function(basis, 6, 16))
	plt.gca().set_box_aspect(1)

	plt.show()