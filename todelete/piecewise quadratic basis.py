import numpy as np
import matplotlib.pyplot as plt

h=1
k = 21/2 #selected node in Nl_star

n_l = np.arange(0.0, 20.5, 0.5)  #Node set
nl = list(n_l)
n_l1 = list(range(0,21,h))  #Reduced node set

n_star = list(set(n_l) - set(n_l1))

N_l = np.array(n_l)
N_l1 = np.array(n_l1)
N_star = np.array(n_star)

m = N_l.size
n = N_l1.size
mstar = N_star.size


#Build mass matrix
M = np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        if i==j:
            if i%2==0:
                M[i][i] = 8/15
            elif i%2==1:
                M[i][i] = 16/15
        elif j==i+1 or i==j+1:
            M[i][j] = 2/15
        elif j==i+2 or i==j+2:
            M[i][j] = -1/15

M[0,0] = 4/15
M[n-1,n-1] = 4/15

#Load b
B=np.zeros((n,mstar))
for i in range(0,n):
    for j in range(0,mstar):
        if j%2==0:
            B[j][j] = 4/15
            B[j+1][j] = 7/15
            B[j+2][j] = -1/15
        elif j%2==1:
            B[j-1][j] = -1/15
            B[j][j] = 7/15
            B[j+1][j] = 4/15

A = np.linalg.solve(M,B)

print(A)

Q_phi = np.zeros((m,mstar))
counter = 0
for i in range(0,m,2):
    Q_phi[i,:] = A[counter]
    counter = counter+1

print(Q_phi)

Phi_matrix = np.identity(m)

index = n_star.index(k)
nindex = nl.index(k)

print(index)
print(nindex)
Psi = Phi_matrix[:,nindex] - Q_phi[:,index]
print(Psi)

plt.plot(N_l,Psi)
plt.show()