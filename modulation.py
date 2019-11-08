# TODO: Set the limits of X and Y axis, depending on the modulation selected.
# TODO: Set colors. Black for base colors (baseline) and other colors for predictions
# TODO: set the received points (array of points for drawing the constellation)
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

        base_constellation_points = np.array([[-3,-3],[-3, -1], [-3, 1], [-3, 3],
                        [-1,-3],[-1, -1], [-1, 1], [-1, 3],
                        [1,-3],[1, -1], [1, 1], [1, 3],
                        [3,-3],[3, -1], [3, 1], [3, 3], ])

        points = np.array([[-10,-10],[-10, -10], [-3, 1], [-3, 3],
                        [-1,-3],[-1, -1], [-0.5, 0.5], [0, 0],
                        [1,-3],[1, -1], [1, 1], [1, 3],
                        [3,-3],[3, -1], [3, 1], [3, 3], ])

        # Create data
        N = len(base_constellation_points)
        area = np.pi*7
        color_base = (0,1,0)
        # colors = np.random.rand(N)
        # color_points = (0,1,0)
        
        # set the limits of the constellation axes
        axes = plt.gca() # gca = get current axes
        axes_limit = self.k + round(self.k*0.2)
        axes.set_xlim([- axes_limit, axes_limit])
        axes.set_ylim([- axes_limit, axes_limit])

        # To add other points than the base ones, just copy next line and change the input.
        plt.scatter(base_constellation_points[:,0],
                    base_constellation_points[:,1],
                    s=area, 
                    c='black',
                    alpha=0.5,
                    marker='*',
                    label= 'baseline points')
        axes.legend()
        plt.title('{}-QAM  Constellation'.format(self.M))
        plt.xlabel('x')
        plt.ylabel('y')
        st.pyplot()
        # without using streamlit:
        # plt.show()

    
if __name__ == "__main__":
    qam = Modulation(16)
    qam.plot_constellation()