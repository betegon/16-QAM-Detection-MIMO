import matplotlib as plt
import numpy as np
import math


def awgn(signal,snr, seed=False):
    """ Generate Aditive White Gaussian Noise (AWGN) from an input signal.

    Args:
        signal (list(float)): Input signal. e.g. signal = [1, 2.3, -1]
        snr          (float): Signal-to-noise ratio in dB.
        seed          (bool): Set seed to reproduce results (good for debugging).
                              Defaults to False.

    Returns:
        noise (list(float)): AWGN generated from an input signal.

    """
    if seed: np.random.seed(1)
    sigpower=sum([math.pow(abs(signal[i]),2) for i in range(len(signal))])
    sigpower=sigpower/len(signal)
    noisepower=sigpower/(math.pow(10,snr/10))
    noise=math.sqrt(noisepower)*(np.random.uniform(-1,1,size=len(signal)))
    return noise
    