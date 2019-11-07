# TODO: Set the limits of X and Y axis, depending modulation
# TODO: Set colors. Black for base colors (baseline) and other colors for predictions

import streamlit as st
import pandas as pd
import numpy as np 
import math
import matplotlib.pyplot as plt

class Modulation:
    def __init__(self,M):
        self.M = M # Size of signal constellation
        self.k = int(np.log2(M)) # Number of bits per symbol 
        # nbits = 3000        # Numer of bits to process
        # numSamplesPerSymbol = 1    # Oversampling factor
    def plot_constellation(self):

        data = np.array([[-3,-3],[-3, -1], [-3, 1], [-3, 3],
                        [-1,-3],[-1, -1], [-1, 1], [-1, 3],
                        [1,-3],[1, -1], [1, 1], [1, 3],
                        [3,-3],[3, -1], [3, 1], [3, 3], ])

        # Create data
        N = len(data)
        color_base = (0,1,0)
        colors = np.random.rand(N)
        color_points = (0,1,0)
        area = np.pi*3

        axes = plt.gca()
        axes.set_xlim([-10, 10])
        axes.set_ylim([-10, 10])

        # Plot
        plt.scatter(data[:,0],data[:,1], s=area, c=color_base, alpha=0.5)

        plt.title('QAM - Constellation')
        plt.xlabel('x')
        plt.ylabel('y')
        st.pyplot()
        # without using streamlit:
        # plt.show()

    
if __name__ == "__main__":
    qam = Modulation(16)
    qam.plot_constellation()