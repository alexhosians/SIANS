# Surface Integral Asteroid $N$-Body Simulator
Welcome to the Surface Integral Asteroid $N$-Body Simulator (SIANS). This code is mainly aimed to solve the full two-body problem, where the asteroids are modelled as ellipsoidal and/or polyhedral shapes, but is also generalized as an $N$-body code.

The mutual potential is calculated with the use of surface integrals outlined by Conway 2016. The advantage of this approach is that the resulting force, torque and mutual potential energy is exact for bodies of ellipsoidal shapes. Furthermore, this method does not suffer from divergences if the bodies are close (but not intersecting).

The following programs and libraries are required for the program to run
* [Python >3.6](https://www.python.org/) - Main programing framework
* [Cython](https://cython.org/) - Provides python wrappers around C code
* [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/) - Numerical library for C

The program is written in both Python 3.6 (or greater) and Cython. It is assumed that you have installed most of the basic libraries, such as numpy and matplotlib. For more details on how the program is used or installation of software, please refer to the documentation.

If you make use of this software for your research, we ask you to cite the following paper:

Alex Ho, Margrethe Wold, Mohammad Poursina, John T. Conway, "The accuracy of mutual potential approximations in simulations of binary asteroids", A&A, 671, A38 (2023), [https://doi.org/10.1051/0004-6361/202245552](https://doi.org/10.1051/0004-6361/202245552)