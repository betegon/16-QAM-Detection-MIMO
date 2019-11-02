import logging
import numpy as np  

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

# Symbols transmitted
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

