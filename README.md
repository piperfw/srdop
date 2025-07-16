# srdop
Spatially resolved dynamics of organic polaritons

gmanp.py
--------
Class containing methods to construct and work with lambda basis defined in
thesis. Called by c2HTC.py / does not need to be edited.
- Note saves calculated structure tensors to .pkl files in ./data/tensors to
  avoid recomputing on later runs

c2HTC.py
--------
Main file for calculating dynamics using second-order cumulants.
Basic example is run using ./c2HTC.py (or python c2HTC.py).

cumulant_in_code.pdf
--------------------
Summary of notations for cumulant equations and coefficients used in the code
(which isn't in the thesis explicitly).

real_space_tb.py
----------------
Code for 1D model solved with 2nd-order cumulants in real space (new main file)


real_space_2d_tb.py
-------------------
Code for 2D model solved with 2nd-order cumulants in real space (experimental)


mean-field_tb_1d.py
----------------
Code for 1D model solved with mean-field in real space (experimental)

mean-field_tb_2d.py
-------------------
Code for 2D model solved with mean-field in real space (experimental)

dispersion.py
-------------
Code to plot 1D plasmonic chain dispersion model from Downing et al. 2018 (J.
Phys. Condens. Matter)

Requirements
------------
- python>=3.11
- numpy, scipy, opt_einsum, matplotlib, mpmath, pretty_traceback (optional),
  progressbar, sparse

