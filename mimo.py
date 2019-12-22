# TODO: 3. Pensar que meter y que no en el bucle
# TODO: 4. Pensar como hacer los bucles (invertidos para poder continuar training cuando queramos)
# TODO: 5. Pensar como ir guardando los resultados en un fichero
# TODO: 6. Mover las funciones a utils o a donde corresponda.
# TODO: 7. Refactor loops for maps and zips.(idiomatic pythonic code ;)
import cvxpy as cp
import math
import numpy as np

from optimization import min_tr_WA
from utils import awgn, gen_symbols, mapping, nearestPD, quantiz


M = 16 # 16-QAM
n = int(np.log2(M)) # Number of bits per symbol
k = 3 # 2 antennas
nbits = 3*k*n # number of bits. minimum k*n
symbols = gen_symbols(nbits,n)
print("Total symbols\n",symbols)
print("\n")
print("SYMBOLS QUE COGE",)
print(symbols[0:2,:])
print("\n")


s = gen_symbols(nbits,n)
print("S",s)
a = np.expand_dims(s[:k,0],-1)
b = np.expand_dims(s[:k,1],-1)
print(a.shape)
print(b.shape)
# s = np.expand_dims(np.stack((s[:,0].transpose,s[:,1].transpose)),-1)
s = np.concatenate((a,b))
print("s ya movida",s.shape)
print("s\n",s)
t = s**2
print(t)

# GENERATE CHANNEL MATRIX
# H is CN(0,1)
mu = np.zeros(2*k)
sigma = np.ones((k,k*2))

H = np.random.normal(loc=mu, scale=sigma, size=(k,k*2)).view(np.complex128)
print(H)
a = np.column_stack((np.real(H), -np.imag(H)))
print("Fila1: \n",a,"\n")
b = np.column_stack((np.imag(H), np.real(H)))
print("Fila2\n",b,"\n")
H_expanded = np.concatenate((a,b), axis=0)
print("Matriz H expandida\n",H_expanded)


# %%
# y = H*s+w;
mu = np.zeros(k)
sigma = np.ones((k,k))
snr = 10

noise = awgn(s,snr)
# noise = np.random.normal(loc=mu, scale=sigma, size=(k,k)).view(np.complex128)
# noise = np.concatenate((np.real(noise), np.imag(noise)), axis=0)

y = H_expanded.dot(s) + noise

print("\nH\n",H_expanded)
print("\ns\n",s)
# print("\nnoise\n",noise)
print("\nnoise \n", noise)
print("\ny\n",y)

# Calculate A, used for min(tr(W*A))
A11 = H_expanded.T.dot(H_expanded) # Check if it is really doing the transpose of H_expanded
A12 = np.zeros((2*k,2*k))
A13 = -H_expanded.T.dot(y) # Check if it is really doing the transpose of H_expanded
A1X = np.column_stack((np.column_stack((A11, A12)), A13))
print("\nFILA 1")
print("\n",A1X)

print("\nFILA 2")
A2X = np.zeros((2*k,A1X.shape[1]))
print("\n",A2X)

print("\nFILA 3")
A31 = (-y.T).dot(H_expanded)
A32= np.zeros((1,2*k))
A33 = (y.T).dot(y)
A3X = np.column_stack((np.column_stack((A31, A32)), A33))
A3X = np.column_stack((A31, A32, A33))
print("\n",A3X)

A = np.concatenate((A1X,A2X,A3X))
print("\n Matriz A")
print("\n",A)
print("\n",A.shape)

W = min_tr_WA(k,A)
W11=W[0:2*k,0:2*k]
print("Esto es W11:",W11)
print("\n")
W13=W[0:2*k,4*k]
print("Estos es W13:",W13)
print("\n")
W31=W[4*k,0:2*k]
print("Estos es W31:",W31)
print("\n")
W_1113=np.column_stack((W11,W13))
print("Column stack",W_1113)
one=np.append(W31,1)
one = np.array([one])
print("\n")
print(one)
print("\n")
print(W_1113.shape)
print(one.shape)
W_ED = np.concatenate((W_1113, one))
print("W_ED\n",W_ED)

print("SIMPLE QUANTIZATION\n\n")
#simple quantization
valores=W[0:2*k,4*k]
simple_quantiz = quantiz(valores,s)
print("quantiz simple niño",simple_quantiz.shape)

# %%
#eigenvalue descomposition
u,S,v=np.linalg.svd(W_ED)#este te calcula svd directamente
print (W_ED)
print ("\n")
print(v)
print ("\n")
print(v[0,0:2*k])
print ("\n")
print(v[0,2*k])
eigen=v[0,0:2*k]/v[0,2*k]
eigen_trans=eigen.transpose()
print("\n")
print("ESto es el cociente:",eigen)
print("\n")
simb=[-3,-1, 1 ,3]
eigen_descomposition = quantiz(eigen_trans,simb)

try:
    v= np.linalg.cholesky(W_ED)
    print("Es positiva")
except np.linalg.LinAlgError:
    v = nearestPD(W_ED)
    print("No Es positiva")
V=v.transpose()
print(V)
print("\n")
r=np.random.random((2*k+1,1))
print("Esto es r:",r)
Atransponer=V[:,0:2*k]
a=Atransponer.transpose()
print("\n")
cosas=np.dot(a,r)
print("\n")
Btransponer=V[:,2*k]
b=Btransponer.transpose()
cosas1=np.dot(b,r)
print("\n")
Aquantiz=(cosas)/(cosas1)
print("Esto es Aquantiz:",Aquantiz)
print("\n")
randomization=quantiz(Aquantiz,simb)
# print(randomization)

# %%
print("\nValor de los símbolos de entrada \n",s)
print("\nValor simple quantization \n", simple_quantiz)
print("\nValor eigenvalue descomposition \n",eigen_descomposition)
print("Valor randomization \n",randomization)
if np.array_equal(s,simple_quantiz) and np.array_equal(s, eigen_descomposition) and np.array_equal(s,randomization):
    print("all good")
else:
    print("simple quatiz bad: ", np.sum(s != simple_quantiz))
    print("Eigen descomposition bad: ",np.sum(s != eigen_descomposition))
    print("Randomization bad: ",np.sum(s != randomization))
    print("no good")
# %%
