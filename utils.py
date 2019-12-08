import math
import matplotlib as plt
import numpy as np


def awgn(signal,snr, seed=False):
    """ Generate Aditive White Gaussian Noise (AWGN) from an input signal.

    Args:
        signal  (np.array): Input signal. e.g. signal = np.array([[1, 2.3, -1]])
        snr        (float): Signal-to-noise ratio in dB.
        seed        (bool): Set seed to reproduce results (good for debugging).
                            Defaults to False.

    Returns:
        noise (np.array): AWGN generated from an input signal.

    """
    if seed: np.random.seed(1)
    sigpower = sum([ math.pow(abs(signal[i]),2) for i in range(len(signal)) ])
    sigpower = sigpower / len(signal)
    noisepower = sigpower / (math.pow(10,snr/10))
    noise = np.random.normal(0, np.sqrt(noisepower), len(signal))
    return np.array([noise]).T


def mapping (decimal_stream):
    """ Mapping decimal numbers to 16-QAM constellation signal.

    Args:
        decimal_stream  (np.array): Decimal numbers to map. 

    Returns:
        input_signal (np.array): Signal to transmit.

    """
    map = {
     "0": [-3,-3],  "1": [-3,-1],  "2": [-3, 3],  "3": [-3, 1],
     "4": [-1,-3],  "5": [-1,-1],  "6": [-1, 3],  "7": [-1, 1],
     "8": [ 3,-3],  "9": [ 3,-1], "10": [ 3, 3], "11": [ 3, 1],
    "12": [ 1,-3], "13": [ 1,-1], "14": [ 1, 3], "15": [ 1, 1]
    }

    input_signal = np.empty(shape=(len(decimal_stream),2))
    for i in range(len(decimal_stream)):
        for key,value in map.items():
            if decimal_stream[i] == int(key):
                # print("value:   {}".format(value))    
                # print("signal:  {}".format(input_signal))
                # print("sgn[0]:  {}".format(input_signal[0]))
                input_signal[i] = value
 
    return input_signal
