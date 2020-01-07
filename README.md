# 16-QAM-Detection-MIMO



## DEPENDENCIES

`Python 3.7.5`

`Numpy 1.17.2`

`matplotlib 3.1.1`

`CVXPY 1.0.25`

## INSTALLATION

Easiest way is to create a new Anaconda environment and install all necessary dependencies.
If prefer not using Anaconda, just install all dependencies using plain `pip`, the _Package Installer for Python_.

### ANACONDA 

`conda create -n mimo python=3.7 matplotlib numpy cvxpy`

`conda activate mimo`

### MOSEK SOLVER

Solver used for optimization is MOSEK. In order to use it, you will need to request  a [mosek](www.mosek.com) trial license and follow the steps indicated there.


## FILES
`mimo.py` 				   - Looping through all SNRs specified for obtaining symbol errors using SDR and  simple quantization, eigenvalue decomposition and randomization approximation techniques.

`plot.py`                   - Calculate Symbol error rates from `mimo.py` output log file and plot them. 

`optimization.py`   - Optimization module, it performs the SDR, minimizing the trace(W*A)

`utils.py`  			   - Diverse utils necessary to perform the MIMO detection.

`detection.py`         - Principal detection module.

`remove_comments`   - Script to remove comments from files.



### REMOVE COMMENTS
As the repository being a mere illustration of MIMO detection, it is full of comments to help grasp all the information.

In order to remove comments from files, just run the script 	`remove_comments.py`. By default is set to remove comments from `detection.py`, creating a new file called `detection_no_comments.py`.

To delete comments from other files, just change the last line of `remove_comments.py`:

```python
remove_comments('detection.py', 'detection_no_comments.py')
```

* Change `detection.py` to the file you want to remove comments (specify path if is outside of root directory) 

* Change `detection_no_comments.py` to your output-No-comments-file that will be generated.

## TODO
* Add code license
* Add support for *Maximum likelihood*, *zeroforcing* and other detectors to compare results with SDR.
* Create an object `Detector` to choose between different detectors implemented.

## REFERENCES
**[1]** _Semidefinite Relaxation for Detection of 16-QAM Signaling in MIMO Channels_ -- -- A. Wiesel ; Y.C. Eldar ; S. Shamai

**[2]** [MOSEK](mosek.com)

**[3]** [CVXPY](www.cvxpy.org)

**[4]** [Mathworks - Compute BER for a QAM system with AWGN](https://www.mathworks.com/help/comm/gs/compute-ber-for-a-qam-system-with-awgn-using-matlab.html
)

**[5]** [Google Colab](https://colab.research.google.com)



