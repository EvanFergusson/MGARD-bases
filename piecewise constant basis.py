import numpy as np
import matplotlib.pyplot as plt

h=2
k = 9 #selected node in Nl_star

n_l = list(range(0,20))  #Node set
n_l1 = list(range(0,20,h))  #Reduced node set

n_star = list(set(n_l) - set(n_l1))

N_l = np.array(n_l)
N_l1 = np.array(n_l1)
N_star = np.array(n_star)

m = N_l.size
n = N_l1.size
mstar = N_star.size


def phi(x):
    return np.full(x.shape, 1)
t1 = np.arange(2.0,4.0,0.1)

def phi2(x):
    return np.full(x.shape, 1)
t2 = np.arange(10.0,14.0,0.1)

#plt.plot(t1, phi(t1))
#plt.plot(t2, phi(t2))
#plt.show()


#Build mass matrix
M = np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        M[i][i] = 2*h
        if j==i-1:
            M[i][j] = h
        elif j==i+1:
            M[i][j] = h

M[0,0] = 2
M[n-1,n-1] = 2

print(M)

#Load b
B=np.zeros((n,mstar))
for i in range(0,n):
    for j in range(0,mstar):
        if j == i:
            B[i][j] = 2
        elif j == i-1:
            B[i][j] = 2

print(B)

A = np.linalg.solve(B,M)
print(A)

Q_phi = np.zeros((m,mstar))
counter = 0
for i in N_l1:
    Q_phi[i,:] = A[counter]
    counter = counter + 1

Phi_matrix = np.zeros((m,m))
for i in range(0,m):
    for j in range(0,m):
        if i==j:
            Phi_matrix[i,j] = 1
        elif j==i+1 or j==i-1:
            Phi_matrix[i,j] = 1

plt.plot(N_l,Phi_matrix[:,k],'.k')

index = n_star.index(k)
Psi = Phi_matrix[:,k] - Q_phi[:,index]

plt.plot(N_l,Psi)
plt.show()