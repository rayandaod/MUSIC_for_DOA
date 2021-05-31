# SSDP mini-project: MUSIC for DOA

## Summary

MUSIC for DOA is a popular algorithm used to obtain the Direction Of Arrival estimation for signals reaching sensor arrays using the MUltiple Signal Classification method.<br>
In ths project, we were asked to implement a **BASIC** and an **IMPROVED** version of the algorithm and apply it on **GENERATED** and **REAL** data, composed of **CORRELATED** and **UNCORRELATED** signals.

As a reference, we were given the project of [Huang Guyue, Iriarte Sainz Diego Gabriel and Nyambuu Lkham](https://github.com/rayandaod/SSDP_mini-project/tree/master/DOA_final).

## Organisation

The project is divided in 3 jupyter notebooks (and a "helper" file):
- MUSIC for DOA on generated data, where we focus on the implementations of both the *basic* and the *improved* version of the MUSIC for DOA algo applied on *generated* data. The major improvements of the second version of the algorithm are the better handling of *correlated* data and the error estimation/correction applied on the microphones array.
- MUSIC for DOA on real data, where we try to adapt the *basic* and *improved* algorithm to work on real samples of signals (mainly sweep and speech).
- Simulations, where we do additional examples and tuning of parameters to try to understand in depth the impact of each components of the algorithm.

## Dependencies

The required librairies to run the project are the following:

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

## Authors

- Rayan Daod Nathoo
- Zewei Xu
- Pierre Gabioud

## References

- Honghao Tang, <em>"DOA estimation based on MUSIC algorithm"</em>, Linnéuniversitetet, 2014
- DAI Zeyang, DU Yuming, <em>"DOA Estimation Based on Improved MUSIC Algorithm"</em>, Chengdu University of Information Technology, 2009
- A. Paulraj, B. Ottersten, R. Roy, A. Swindlehurst, G. Xu and T. Kailath, <em>"Subspace Methods for Directions-of-Arrival Estimation"</em>, Handbook of Statistics, 1993
- M. Devendra and K. Manjunathachari, <em>"DOA estimation of a system using MUSIC method"</em>, International Conference on Signal Processing and Communication Engineering Systems, 2015, pp. 309-313, doi: 10.1109/SPACES.2015.7058272.
- Ahmad, Mushtaq & Zhang, Xiaofei, <em>"Performance of Music Algorithm for DOA Estimation"</em>, 2016

