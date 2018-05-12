/* McKernel: A Library for Approximate Kernel Expansions in Log-linear Time 
   Curtó, Zarza, Yang, Smola, De La Torre, Ngo, and Van Gool 		    

   Authors: Curtó and Zarza
   {curto,zarza}@tinet.cat 						    */		             				     
					             				     
README

To run the examples, enter to the given folders and run make as follows:
		$ cd /examples/fwht	
		$ make
		$ ./FWHT

To include the library in your code, follow the next instructions or see the reference make files.	 
							 
Use g++-4.8 or above versions and activate the next flags:

	-msse2 (-mavx) -O3 -std=c++11 -USE_SSE2

add #include "hpp/Factory_McKernel.hpp" in your test file.

RBF Gaussian, here is a compilation example:

	$ g++-4.8 -o Factory_McKernel examples/RBF_Gaussian_Test.cpp src/Factory_McKernel.cpp src/hash.cpp -msse2 -mavx -O3 -std=c++11 -USE_SSE2
	$ ./Factory_McKernel

RBF Matern, here is a compilation example:

	$ g++-4.8 -o Factory_McKernel examples/RBF_Matern_Test.cpp src/Factory_McKernel.cpp src/hash.cpp -msse2 -mavx -O3 -std=c++11 -USE_SSE2
	$ ./Factory_McKernel

If you want to use FWHT, add #include "hpp/McKernel.hpp" in your test file and compile with flags 

	-msse2 (-mavx) -O3 -std=c++11

FWHT, here is a compilation example:

	$ g++-4.8 -o FWHT_Test FWHT_Test.cpp -msse2 -mavx -O3 -std=c++11
	$ ./FWHT_Test


*Note: if -mavx is not compatible with your computer, please remove the flag.
