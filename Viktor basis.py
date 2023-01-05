import numpy as np
import matplotlib.pyplot as plt

h=2
k = 7 #selected node in Nl_star

n_l = list(range(0,20))  #Node set
n_l1 = list(range(0,20,h))  #Reduced node set

print(n_l)
print(n_l1)

n_star = list(set(n_l) - set(n_l1))

print(n_star)

N_l = np.array(n_l)
N_l1 = np.array(n_l1)
N_star = np.array(n_star)

m = N_l.size
n = N_l1.size
mstar = N_star.size

i = 16
phi = np.zeros(m)
for j in N_l:
    if j == i:
        phi[j] = 1
    elif j==i-1 or j == i+1:
        phi[j] = 1/2
    else:
        phi[j] = 0

i2 = 15
phi2 = np.zeros(m)
for j in range(0,m):
    if j == i2:
        phi2[j] = 1
    else:
        phi2[j] = 0


plt.plot(N_l,phi)
plt.plot(N_l,phi2)
plt.show()

#Build mass matrix
M = np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        M[i][i] = (2*h)/3
        if j==i-1:
            M[i][j] = h/6
        elif j==i+1:
            M[i][j] = h/6

print(M)

#Load b
B=np.zeros((n,mstar))
for i in range(0,n):
    for j in range(0,mstar):
        if j == i:
            B[i][j] = 1/2
        elif j == i-1:
            B[i][j] = 1/2

print(B)

A = np.linalg.solve(M,B)

print(A)

Q_phi = np.zeros((m,mstar))
counter = 0
for i in N_l1:
    Q_phi[i,:] = A[counter]
    counter = counter + 1

print(Q_phi)

Phi_matrix = np.identity(m)

index = n_star.index(k)
print(index)
Psi = Phi_matrix[:,k] - Q_phi[:,index]
print(Psi)

plt.plot(N_l,Psi)
plt.show()