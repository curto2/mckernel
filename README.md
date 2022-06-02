# McKernel

McKernel: A Library for Approximate Kernel Expansions in Log-linear Time.

<p align="center">
<img src="fwh.png" width="400">
<img src="rbfmatern.png" width="400">
</p>

In order to reproduce the code, please enter the folder lg, sdd+ or standard and follow the instructions on the README. For compilation it is only necessary to invoke make (and therefore run the makefile) on the given example folder and then run the executable. For an example on how to run the experiments, see the capsule on Code Ocean ([doi.org/10.24433/CO.3851581.v1](https://doi.org/10.24433/CO.3851581.v1)) where MNIST and FASHION-MNIST have already been loaded and a main script to compile and test all examples is given; output of the compilation can be seen on the folder results.

--------------------------------------------------------
Abstract
--------------------------------------------------------
The library explores the applicability of the Hadamard as an input modulator for problems of classification. It
introduces a framework in C++ to use kernel approximates in the mini-batch setting with Stochastic Gradient
Descent. The algorithm requires to compute the product of matrices Walsh Hadamard. A free-standing cache
friendly Fast Walsh Hadamard that achieves compelling speed is provided, as well as a lightweight efficient CPU
implementation of the method for research and practical purposes alike.

--------------------------------------------------------
Dependencies
--------------------------------------------------------
McKernel is written entirely in C++ with no additional prerequisites than a minimal setup on Ubuntu, which is ideal for developing applications on devices that require little memory footprint and minimal codebase such as embedded systems or unmanned aerial vehicles.

--------------------------------------------------------
Citation
--------------------------------------------------------

J. de Curtò, I. de Zarzà, Hong Yan and Carlos T. Calafate. On the applicability of the Hadamard as an input modulator for problems of classification. Software Impacts. 2022.

--------------------------------------------------------
Change Log
--------------------------------------------------------

Version 2.2, released on 06/06/2019.

Version 2.1, released on 04/06/2019.

Version 2.0, released on 26/03/2019.

Version 1.1, released on 24/01/2019.

Version 1.0, released on 12/05/2018.

--------------------------------------------------------
File Information
--------------------------------------------------------

- Standard (mckernel/standard).
  - Library McKernel.
- Standard+ (mckernel/sdd+).
  - Library McKernel. Pseudo-random numbers generated with functions of hashing. Suitable for distributed applications. Recommended.
- Learning (mckernel/lg).
  - DL framework to reproduce experiments in the paper.

--------------------------------------------------------
Documentation (automatically generated with Doxygen)
--------------------------------------------------------
- McKernel (documentation/mckernel/html/index.html).
- lg (documentation/lg/html/index.html).

--------------------------------------------------------
Acknowledgements
--------------------------------------------------------
The authors are with the [Research Group on Unmanned Aerial Vehicles](https://grc.webs.upv.es/members/default.html) at Universitat Politècnica de València, the [Department of Electrical Engineering](https://www.ee.cityu.edu.hk/) at City University of Hong Kong and with the [Centre for Intelligent Multidimensional Data Analysis](https://www.innocimda.com/), a private research center at City University of Hong Kong. 

This work is supported by HK Innovation and Technology Commission (InnoHK Project CIMDA) and HK Research Grants Council (Project CityU 11204821).

--------------------------------------------------------
Contact
--------------------------------------------------------
Code is mantained by J. de Curtò (decurto@doctor.upv.es) and I. de Zarzà (dezarza@doctor.upv.es); doctoral students in Computer Science at the aforementioned institution.
