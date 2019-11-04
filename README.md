# 16-QAM-Detection-MIMO



## DEPENDENCIES

`Python 3.7.5`

`Numpy 1.17.2`

`logging 0.5.1.2`

## INSTALLATION

Easiest way is to create a new Anaconda environment and install all necesary dependencies.
If prefer not using Anaconda, just install all dependencies using plain `pip`, the _Package Installer for Python_.

### ANACONDA 

`conda create -n mimo python=3.7 numpy`

Install `streamlit` using pip, as it wont be available from anaconda current channels (nov/2019).

`pip install streamlit`

`conda activate mimo`


## FILES
`detection.py`      - Principal detection module.

`remove_comments`   - Script to remove comments from files.

## USAGE

### REMOVE COMMENTS
As the repository being a mere illustration of MIMO detection, it is full of comments to help grasp all the information.

In order to remove comments from files, just run the script `remove_comments.py`. By default is set to remove comments from `detection.py`, creating a new file called `detection_no_comments.py`.

To delete comments from other files, just change the last line of `remove_comments.py`:

```python
remove_comments('detection.py', 'detection_no_comments.py')
```

* Change `detection.py` to the file you want to remove comments (specify path if is outside of root directory) 

* Change `detection_no_comments.py` to your output-No-comments-file that will be generated.

## TODO
* Check all dependencies finally used (plt??)
* Set parameters in Streamlit
* Create object detector to choose between different detectors implemented.

## REFERENCES
**[1]** _Semidefinite Relaxation for Detection of 16-QAM Signaling in MIMO Channels_ -- -- A. Wiesel ; Y.C. Eldar ; S. Shamai

**[2]** [Streamlit](https://streamlit.io/)



