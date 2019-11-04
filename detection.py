import logging
import numpy as np  
import streamlit as st
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
nbits = 3000        # Numer of bits to proceess
numSamplesPerSymbol = 1    # Oversampling factor
EbNo = 10           # arbitrarily set to 10 dB
logger.info("Parameter definition: \n   M = {}\n   k = {}\n   n = {}\n   numSamplesPerSymbol= {}\n"
            .format(M, k, nbits, numSamplesPerSymbol))

# Create binary data stream
binaryDataStream = np.random.randint(2, size=(int(nbits/k),k))
logger.debug("binary DataStream:\n{}".format(binaryDataStream))

st.subheader('Binary data Stream')
st.write(binaryDataStream)
decDataStream = binaryDataStream.dot(1 << np.arange(binaryDataStream.shape[-1] -1, -1,-1))

st.subheader('# integer symbols transmitted')
hist_values = np.histogram(decDataStream, bins=M, range=(0,M))[0]
st.bar_chart(hist_values)   


st.subheader('Adding White Gaussian Noise')




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