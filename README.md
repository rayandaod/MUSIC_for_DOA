# MUSIC for DOA
Mini-project for the EPFL Master course [COM 500 - Statistical Signal Data Processing through Applications](https://edu.epfl.ch/coursebook/en/statistical-signal-and-data-processing-through-applications-COM-500)
## Summary

MUSIC (MUltiple Signal Classification) is a popular algorithm used in different fields but in particular to obtain the Direction Of Arrival when receiving signals on a multiple sensor array, and under some assumptions.<br>
In this project, we were asked to implement a **BASIC** and an **IMPROVED** version of the algorithm and to apply it on **GENERATED** and **REAL** data, composed of **CORRELATED** and **UNCORRELATED** signals.

The real data was provided by the course team, and was also retrieved from the [Pyramic dataset](https://github.com/fakufaku/pyramic-dataset).

As a reference, we were given the previous submission of [Huang Guyue, Iriarte Sainz Diego Gabriel and Nyambuu Lkham](https://github.com/rayandaod/SSDP_mini-project/tree/master/DOA_final).

## Project architecture
```
.
├── data/                             # Data folder - used in the real_data.ipynb Jupyter notebook
├── instructions/                     # Instructions given by the course team
├── notebooks/                        # Source files (described below)
    ├── generated_data.ipynb
    ├── real_data.ipynb
    ├── simulations.ipynb
    ├── helper.py
├── references/                       # Useful and interesting papers on the subject
├── .gitignore                        
├── README.md
└── requirements.txt
```

The ```notebooks``` folder is divided in 3 jupyter notebooks, and one python file:
- [generated_data.ipynb](https://github.com/rayandaod/SSDP_mini-project/blob/master/notebooks/generated_data.ipynb) - implementation of both the *basic* and the *improved* version of the MUSIC for DOA algorithm applied on *generated* data. The major improvements of the second version of the algorithm are the better handling of *correlated* data and the error estimation/correction applied on the microphones array.
- [real_data.ipynb](https://github.com/rayandaod/SSDP_mini-project/blob/master/notebooks/real_data.ipynb) - adapt the *basic* and *improved* algorithm to work on real recordings (mainly sweep and speech) provided by the course team and from the [Pyramic dataset](https://github.com/fakufaku/pyramic-dataset). Implementation of our own time-frame based technique.
- [Simulations.ipynb](https://github.com/rayandaod/SSDP_mini-project/blob/master/notebooks/Simulations.ipynb) - additional examples and tuning of parameters to try to understand in depth the impact of each components of the algorithm.
- [helper.py](https://github.com/rayandaod/SSDP_mini-project/blob/master/notebooks/helper.py) - implementation of helper functions used through the previous notebooks.

## Language and dependencies

Python 3

The required Python librairies to run the project are the following:

```bash
jupyter
numpy
matplotlib
scipy
json
math
datetime
ipython 
```

## Quick start

Clone the repository

```bash
git clone https://github.com/rayandaod/SSDP_mini-project.git
cd SSDP_mini-project
```

Download the content of the data folder [here](https://drive.google.com/drive/folders/1hDV3cjiMLJApw9P14WEQ8ffyV7b0Djv9?usp=sharing) and place in ```SSDP_mini-project/data/```

Create your virtual environment (optional)

Install the dependencies and start a local Jupyter server
```bash
pip install requirements.txt
jupyter notebook
```

Run the [generated_data.ipynb](https://github.com/rayandaod/SSDP_mini-project/blob/master/notebooks/generated_data.ipynb) or the [real_data.ipynb](https://github.com/rayandaod/SSDP_mini-project/blob/master/notebooks/real_data.ipynb) notebook

## Authors

- Rayan Daod Nathoo
- Zewei Xu
- Pierre Gabioud

## Acknowledgements
Thank you [Professor Andrea Ridolfi](https://people.epfl.ch/andrea.ridolfi) and his team for this project!

## References

- Honghao Tang, <em>"DOA estimation based on MUSIC algorithm"</em>, Linnéuniversitetet, 2014
- DAI Zeyang, DU Yuming, <em>"DOA Estimation Based on Improved MUSIC Algorithm"</em>, Chengdu University of Information Technology, 2009
- A. Paulraj, B. Ottersten, R. Roy, A. Swindlehurst, G. Xu and T. Kailath, <em>"Subspace Methods for Directions-of-Arrival Estimation"</em>, Handbook of Statistics, 1993
- M. Devendra and K. Manjunathachari, <em>"DOA estimation of a system using MUSIC method"</em>, International Conference on Signal Processing and Communication Engineering Systems, 2015, pp. 309-313, doi: 10.1109/SPACES.2015.7058272.
- Ahmad, Mushtaq & Zhang, Xiaofei, <em>"Performance of Music Algorithm for DOA Estimation"</em>, 2016
- [Pyramic dataset](https://github.com/fakufaku/pyramic-dataset)
- [Pyroomacoustics library](https://github.com/LCAV/pyroomacoustics)
