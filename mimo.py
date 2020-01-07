import numpy as np
import os

from optimization import min_tr_WA
from utils import awgn, gen_symbols, nearestPD, quantiz

M = 16  # 16-QAM
n = int(np.log2(M))  # Number of bits per symbol
k = 3  # 6 antennas
nbits = 100000*k*n  # number of bits. minimum k*n
symbols = gen_symbols(nbits, n)
snr_array = [8, 10, 12, 14, 16, 18]


# GENERATE CHANNEL MATRIX
# H is CN(0,1)
mu = np.zeros(2*k)
sigma = np.ones((k, k*2))

H = np.random.normal(loc=mu, scale=sigma, size=(k, k*2)).view(np.complex128)
a = np.column_stack((np.real(H), -np.imag(H)))
b = np.column_stack((np.imag(H), np.real(H)))
H_expanded = np.concatenate((a, b), axis=0)

log_file = "log-errors.txt"
if not os.path.exists(log_file):
    os.mknod(log_file)
file1 = open(log_file, "a")

for i in range(int(nbits/(k*n))):
    s_real = np.expand_dims(symbols[i*k:(i+1)*k, 0], -1)
    s_imag = np.expand_dims(symbols[i*k:(i+1)*k, 1], -1)
    s = np.concatenate((s_real, s_imag))

    for snr in snr_array:

        noise = awgn(s, snr)
        y = H_expanded.dot(s) + noise

        # Calculate A, used for min(tr(W*A))
        A11 = H_expanded.T.dot(H_expanded)
        A12 = np.zeros((2*k, 2*k))
        A13 = -H_expanded.T.dot(y)
        A1X = np.column_stack((np.column_stack((A11, A12)), A13))
        A2X = np.zeros((2*k, A1X.shape[1]))
        A31 = (-y.T).dot(H_expanded)
        A32 = np.zeros((1, 2*k))
        A33 = (y.T).dot(y)
        A3X = np.column_stack((np.column_stack((A31, A32)), A33))
        A3X = np.column_stack((A31, A32, A33))
        A = np.concatenate((A1X, A2X, A3X))

        W = min_tr_WA(k, A)

        if W is not None:
            W11 = W[0:2*k, 0:2*k]
            W13 = W[0:2*k, 4*k]
            W31 = W[4*k, 0:2*k]
            W_1113 = np.column_stack((W11, W13))
            one = np.append(W31, 1)
            one = np.array([one])
            W_ED = np.concatenate((W_1113, one))

            # symbols to quantize
            simb = [-3, -1, 1, 3]

            # Simple quantization
            values = W[0:2*k, 4*k]
            simple_quantiz = quantiz(values, simb)

            # Eigenvalue descomposition
            u, S, v = np.linalg.svd(W_ED)
            eigen = v[0, 0:2*k]/v[0, 2*k]
            eigen_trans = eigen.transpose()
            eigen_descomposition = quantiz(eigen_trans, simb)

            # Randomization
            try:
                v = np.linalg.cholesky(W_ED)
            except np.linalg.LinAlgError:
                v = nearestPD(W_ED)
            V = v.transpose()
            r = np.random.random((2*k+1, 1))
            Atotranspose = V[:, 0:2*k]
            a = Atotranspose.transpose()
            cosas = np.dot(a, r)
            Btotranspose = V[:, 2*k]
            b = Btotranspose.transpose()
            cosas1 = np.dot(b, r)
            Aquantiz = (cosas)/(cosas1)
            randomization = quantiz(Aquantiz, simb)
            print("\nSymbols to transmit: \n", s)
            print("\nSimple quantization result:\n", simple_quantiz)
            print("\nEigenvalue descomposition result:\n", eigen_descomposition)
            print("Randomization result:\n", randomization)
            if np.array_equal(s, simple_quantiz) and np.array_equal(s, eigen_descomposition) and np.array_equal(s, randomization):
                file1.write("{},{},{},".format(0, 0, 0))
            else:
                file1.write("{},{},{},".format(
                                        np.sum(s != simple_quantiz),
                                        np.sum(s != eigen_descomposition),
                                        np.sum(s != randomization)))
    file1.write("\n")
    file1.flush()
