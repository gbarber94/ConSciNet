# Joint Parameter Discovery and Generative Modeling of Dynamic Systems

Gregory Barber, Mulugeta A. Haile, and Tzikang Chen | 2021

This repository contains the trained models, data and code used in the publication: ADD PUBLICATION LINK .

![](figures/ConSciNet_arch.png)

![](figures/ConSciNet_pen.png)

___
## Summary

Given an unknown dynamical system, we are often interested in gaining insights into its physical
parameters. For instance, given an observation of the motion of a harmonic oscillator in the form of
video frames or time-series data, we wish to know the stiffness and mass. How
do we do this from observation and without the knowledge of the dynamics model? In this paper,
we present a neural framework for estimating physical parameters in a manner consistent with the
underlying physics. The neural framework uses a deep latent variable model that disentangles the
system’s physical parameters from canonical coordinate observations. The network then returns a
Hamiltonian parameterization that generalizes well with respect to the discovered physical parameters.
We apply our framework to simple dynamical problems and show that it discovers physically
meaningful parameters while respecting the governing physics.

## Colab links:

- Pendulum:    add link when public
- Mass-spring: add link when public

## nbviewer links:
- Pendulum: https://nbviewer.jupyter.org/github.com/gbarber94/ConSciNet/blob/main/ConSciNet_pendulum.ipynb
- Mass-spring: https://nbviewer.jupyter.org/github.com/gbarber94/ConSciNet/blob/main/ConSciNet_mass_spring.ipynb

## Dependencies
- PyTorch
- NumPy
- SciPy
- Autograd
