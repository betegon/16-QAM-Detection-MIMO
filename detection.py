# TODO: Create a left menu and add all variables as: Amplitud (for constellation representation, Eb/N0,...)

import logging
from modulation import Modulation
import numpy as np  
import streamlit as st
from utils import awgn, mapping
st.title('MIMO Detection')

#### #### #### [START logger handler] #### #### #### 
# create logger
logger = logging.getLogger('MIMO detection')
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
#### #### #### [END logger handler] #### #### ####  


# Parameter definition
M = 16              # Size of signal constellation
k = int(np.log2(M)) # Number of bits per symbol
nbits = 50000        # Numer of bits to process
numSamplesPerSymbol = 1    # Oversampling factor
logger.info("Parameter definition: \n   M = {}\n   k = {}\n   n = {}\n   numSamplesPerSymbol= {}\n"
            .format(M, k, nbits, numSamplesPerSymbol))

# Set seed for debugging
if st.checkbox('Set seed for debugging.'): np.random.seed(1)

#### #### #### [START symbols transmitted, x] #### #### #### 
# Create binary data stream
binaryDataStream = np.random.randint(2, size=(int(nbits/k),k))
logger.debug("binary DataStream:\n{}".format(binaryDataStream))

st.subheader('Binary data Stream')
st.write(binaryDataStream)
decDataStream = binaryDataStream.dot(1 << np.arange(binaryDataStream.shape[-1] -1, -1,-1))


# PLot decimal Symbols transmitted (x)
st.subheader('# integer symbols transmitted')
hist_values = np.histogram(decDataStream, bins=M, range=(0,M))[0]
st.bar_chart(hist_values)   
#### #### #### [END symbols transmitted, x] #### #### #### 


#### #### #### [START AWGN] #### #### #### 
st.subheader('Adding White Gaussian Noise')

# Set EbNo
# arbitrarily set the slider in 10 dB 
EbNo = st.slider('Ratio of bit energy to noise power spectral density, Eb/N0:',0,30,10,1)          
# Calculate signal-to-noise ratio
snr = EbNo + 10*np.log10(k) - 10*np.log10(numSamplesPerSymbol)
st.write("Signal-to-noise ratio achieved: ",snr)

# QAM modulation
qam = Modulation(M)
# Mapping 
input_signal = mapping(decDataStream)
st.write(input_signal)

noise_x = awgn(input_signal[:,0].tolist(),snr)
noise_y = awgn(input_signal[:,1].tolist(),snr)
noise = np.column_stack((noise_x,noise_y))
signal_and_noise = input_signal + noise
print(signal_and_noise)

qam.plot_constellation(signal_and_noise)



#### #### #### [END AWGN] #### #### #### 








#### #### #### [START MIMO 2x2 System] #### #### #### 
x = np.array([[1, 1]]).T
logger.debug("Symbols transsmited:\n{}".format(x))

# Channel coefficients
h = np.array([[2, 2], [3, 3]])
logger.debug("Channel coefficients:\n{}".format(h))

# Noise samples
n = np.array([[4, 4]]).T
logger.debug("Noise samples:\n{}".format(n))

# Symbols received
y = h.dot(x) + n
logger.debug("Symbols received:\n{}".format(y))
#### #### #### [END MIMO 2x2 System] #### #### #### 
