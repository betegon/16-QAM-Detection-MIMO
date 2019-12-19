# TODO: 1. conseguir qeu funcione randomsimb() o crearla de otra forma. mirar mi codigo viejo de detection.py donde genero streams aleatorios igual es mejor usar eso para generar la ostia y hacer el bucle
# TODO: 2. Asegurarme de que funciona bien hasta aquí
# TODO: 3. Pensar que meter y que no en el bucle
# TODO: 4. Pensar como hacer los bucles (invertidos para poder continuar training cuando queramos)
# TODO: 5. Pensar como ir guardando los resultados en un fichero
# TODO: 6. Mover las funciones a utils o a donde corresponda.



# %%
'''
<a href="https://colab.research.google.com/github/betegon/16-QAM-Detection-MIMO/blob/master/TAC_MIMO.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
'''

from utils import awgn

# %%
import math
import numpy as np
import cvxpy as cp


# %%
'''
## EJEMPLO CON 2 antenas

'''

# %%
k = 2 # 2 antennas

# %%
# select a random sample without replacement
from random import seed
from random import sample


def randomsimb():
  # seed random number generator
  # prepare a sequence
  sequence = [-3, -1, 1, 3,-3, -1, 1, 3, -3, -1, 1, 3,-3, -1, 1, 3,-3, -1, 1, 3]
  # select a subset without replacement
  subset = sample(sequence, 2*k)
  return np.expand_dims(subset, -1)

# %%
s = randomsimb()
print("forma de S",s.shape)

# s_matrix = np.array([[1, 1],[1, 1]])
# print(s_matrix)
# s = (np.array([s_matrix.flatten('F')])).transpose()
# randomsimb()
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

# %%
## CALCULO DE W (variable a minimizar)
# w = np.concatenate((s,t,[[1]]))
# w_transpose = w.transpose()
# print("w\n",w)
# print("w_transpose\n",w_transpose)

# W = w.dot(w_transpose)
# print("\nBig W\n",W)
# print("\nBig W\n",W.shape)


# %%
# Cálculo de la matriz a multiplicar por H
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

# A = [H'*H zeros(2*N,2*N) -H'*y; zeros(2*N,4*N+1); -y'*H zeros(1,2*N) y'*y];


def min_tr_W(k,A):

    # Problem data.
    np.random.seed(1)
    #k = 2 # number of antennas (TX y RX)
    zero_array = np.zeros(2*k)

    # Construct the problem.
    W = cp.Variable((4*k+1,4*k+1), PSD=True)

    objective = cp.Minimize(cp.trace(cp.matmul(W,A)))
    constraints = [W >> 0,
               cp.diag(W[0:2*k, 0:2*k]) - W[2*k:4*k,4*k] == 0,
               cp.diag(W[2*k:4*k, 2*k:4*k]) - 10*W[2*k:4*k,4*k] + 9*np.ones((2*k)) == 0,
               W[4*k, 4*k] == 1 ]

    # formulate and solve problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # print("Optimal value", prob.solve())
    print("Optimal value of W: ")
    print(W.value) # A numpy ndarray.
    return W.value

W = min_tr_W(k,A)
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

# %%
#quantiz version BT
def quantiz(entry, symbols):
    result = np.empty((len(entry),1))
    for i in range(len(entry)):
        minimum = float("inf")
        for val in symbols:
            if abs(val - entry[i]) < minimum:
                result[i,0] = val
                minimum = abs(val - entry[i])
    return result

# %%
#simple quantization

valores=W[0:2*k,4*k]
simple_quantiz = quantiz(valores,s)
print(simple_quantiz)

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


# %%
from numpy import linalg as la

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

if __name__ == '__main__':
    import numpy as np
    for i in range(10):
        for j in range(2, 100):
            A = np.random.randn(j, j)
            B = nearestPD(A)
            assert(isPD(B))
    print('unit test passed!')

# %%
#randomization
#from scipy.linalg import cholesky
#v2=cholesky(W_ED)
#v=np.linalg.cholesky(W_ED)
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
    print("no good")
# %%
