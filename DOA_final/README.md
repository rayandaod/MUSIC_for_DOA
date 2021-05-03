# Music for Direction of Arrival DOA
**mini-project for COM-500 (Statistical Signal Processing through Applications) at EPFL**

## Structure
The folder contains the following files:

* DOA_part1(generated-data).ipynb
* DOA_part2(real-data).ipynb
* fq_sample3_spkr0_angle20.wav
* fq_sample3_spkr0_angle30.wav
* fq_sample3_spkr0_angle70.wav
* protocol.json
* speech_correlated.wav
* speech_uncorrelated.wav
* sweep_correlated.wav
* sweep_uncorrelated.wav

The two ipynb notebooks contain the demos and the simulations. The first one (DOA_part1(generated-data).ipynb) contains the simulations of Music and Improved Music for generated data (similar to examples presented on reference papers). 
The second one (DOA_part2(real-data).ipynb) contains the simulations of Music and Improved music for real data + provided data, the simulations include both Music and improved Music. The last part of this notebook also contains an implementation with the library pyroomacoustics.
The wav files which start with fq_sample... are required for running the notebook, they are some examples of real data collected from the experiment data set at <https://zenodo.org/record/1209563#.XOWuclMzbyw> and the other wav files are the real data provided for this miniproject.
Protocol.json contains some description of the experiment setup such as microphones and sources arrangement. this information can be found at: <https://github.com/fakufaku/pyramic-dataset>

## Instructions
Each notebook has commented cells and some comments on the code. It is better to start with the first notebook and then follow to the second one, the subtitles also indicate the order to follow (e.g. Part 1, Part 2, Part 3). Run each cell of the notebook to get the resuls
For the last part of the second notebook (real data), you may need to install the library pyroomacoustics:
```sh
$ pip install pyroomacoustics
```


**Authors:**

* *Huang Guyue*
* *Iriarte Sainz Diego Gabriel*
* *Nyambuu Lkham*

