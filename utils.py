# TODO: mapping() to handle multidimensional np.arrays

import math
import numpy as np
from numpy import linalg as la


def awgn(signal, snr, seed=False):
    """ Generate Aditive White Gaussian Noise (AWGN) from an input signal.

    Args:
        signal  (np.array): Input signal e.g. signal = np.array([[1, 2.3, -1]])
        snr        (float): Signal-to-noise ratio in dB.
        seed        (bool): Set seed to reproduce results (good for debugging).
                            Defaults to False.

    Returns:
        noise (np.array): AWGN generated from an input signal.

    """
    if seed:
        np.random.seed(1)
    sigpower = sum([math.pow(abs(signal[i]), 2) for i in range(len(signal))])
    sigpower = sigpower / len(signal)
    noisepower = sigpower / (math.pow(10, snr/10))
    noise = np.random.normal(0, np.sqrt(noisepower), len(signal))
    return np.array([noise]).T


def mapping(decimal_stream):
    """ Mapping decimal numbers to 16-QAM constellation signal.

    Args:
        decimal_stream  (np.array): Decimal numbers to map.

    Returns:
        input_signal (np.array): Signal to transmit.

    """
    map = {
     "0":  [-3, -3],  "1": [-3, -1],  "2": [-3, 3],  "3": [-3, 1],
     "4":  [-1, -3],  "5": [-1, -1],  "6": [-1, 3],  "7": [-1, 1],
     "8":   [3, -3],  "9":  [3, -1], "10":  [3, 3], "11":  [3, 1],
     "12":  [1, -3], "13":  [1, -1], "14":  [1, 3], "15":  [1, 1]
    }

    input_signal = np.empty(shape=(len(decimal_stream), 2))
    for i in range(len(decimal_stream)):
        for key, value in map.items():
            if decimal_stream[i] == int(key):
                # print("value:   {}".format(value))
                # print("signal:  {}".format(input_signal))
                # print("sgn[0]:  {}".format(input_signal[0]))
                input_signal[i] = value
    return input_signal


def gen_symbols(nbits, n):
    """ Generate n^2-QAM symbols

    Args:
        nbits (int): Number of bits to generate.
        n     (int): bits per symbol. e.g. n=4 -> 2^4 = 16QAM

    Returns:
        (np.ndarray): Contains constellation symbols. e.g. [[-1 3],[3, 1]...]

    """
    binaryDataStream = np.random.randint(2, size=(int(nbits/n), n))
    decDataStream = binaryDataStream.dot(1 << np.arange(binaryDataStream.shape[-1] - 1, -1, -1))
    return mapping(decDataStream)


def quantiz(entry, symbols):
    """ Quantize an array from given symbols.

    Args:
        entry   (np.array): Array of float numbers to quantize.
        symbols (np.array): Array of quantization values.

    Returns:
        result (np.array): Array of quantized values.
    """
    result = np.empty((len(entry), 1))
    for i in range(len(entry)):
        minimum = float("inf")
        for val in symbols:
            if abs(val - entry[i]) < minimum:
                result[i, 0] = val
                minimum = abs(val - entry[i])
    return result


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
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually
    # on the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
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
