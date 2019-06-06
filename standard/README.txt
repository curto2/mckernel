/* McKernel: Approximate Kernel Expansions in Log-linear Time through Randomization		    

   Authors: Curtó and Zarza
   {curto,zarza}.2@my.cityu.edu.hk 						    */            				     
                                                                           
_|      _|            _|    _|                                          _|  
_|_|  _|_|    _|_|_|  _|  _|      _|_|    _|  _|_|  _|_|_|      _|_|    _|  
_|  _|  _|  _|        _|_|      _|_|_|_|  _|_|      _|    _|  _|_|_|_|  _|  
_|      _|  _|        _|  _|    _|        _|        _|    _|  _|        _|  
_|      _|    _|_|_|  _|    _|    _|_|_|  _|        _|    _|    _|_|_|  _|
					             				     
README					     					 

To run the examples, enter the given folders and run make as follows:

	$ cd /examples/fwh	
	$ make
	$ ./FWH

To include the library in your code, follow the next instructions or see the reference make files.
							 
Use g++-4.8 or above versions and activate the flags:

	-msse2 (-mavx) -O3 -std=c++11 -USE_SSE2

add #include "hpp/Factory_McKernel.hpp" in your test file.

RBF GAUSSIAN. Here is an example of how to compile the code:

	$ g++-4.8 -o Factory_McKernel examples/RBF_Test.cpp src/Factory_McKernel.cpp -msse2 -mavx -O3 -std=c++11 -USE_SSE2
	$ ./Factory_McKernel

RBF MATÉRN. Here is an example of how to compile the code:

	$ g++-4.8 -o Factory_McKernel examples/RBF_MATERN_Test.cpp src/Factory_McKernel.cpp -msse2 -mavx -O3 -std=c++11 -USE_SSE2
	$ ./Factory_McKernel

If you want to use FWH, add #include "hpp/McKernel.hpp" in your test file and compile with flags 

	-msse2 (-mavx) -O3 -std=c++11

FWH. Here is an example of how to compile the code:

	$ g++-4.8 -o FWH_Test FWH_Test.cpp -msse2 -mavx -O3 -std=c++11
	$ ./FWH_Test


*Note: if -mavx is not compatible with your computer, please remove the flag.
