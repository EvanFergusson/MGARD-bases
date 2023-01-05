import numpy as np
import matplotlib.pyplot as plt

h=2
#k = 7 #selected node in Nl_star

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

