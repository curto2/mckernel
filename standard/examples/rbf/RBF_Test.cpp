/* McKernel: A Library for Approximate Kernel Expansions in Log-linear Time.		    

   Authors: Curtó and Zarza.
   c@decurto.tw z@dezarza.tw 						    */

#include <ctime>
#include <iostream>
#include "../../hpp/Factory_McKernel.hpp"

int main()
{
	printf(R"EOF(                                                                            
_|      _|            _|    _|                                          _|  
_|_|  _|_|    _|_|_|  _|  _|      _|_|    _|  _|_|  _|_|_|      _|_|    _|  
_|  _|  _|  _|        _|_|      _|_|_|_|  _|_|      _|    _|  _|_|_|_|  _|  
_|      _|  _|        _|  _|    _|        _|        _|    _|  _|        _|  
_|      _|    _|_|_|  _|    _|    _|_|_|  _|        _|    _|    _|_|_|  _|

)EOF");

	//Initialize timing counter
	clock_t startTime, endTime;

	//Initialize variables
	unsigned long nv = 1UL << 12;
	unsigned long dn = 512;
	unsigned long D = 2 * dn;
	unsigned long l = nv * dn;
	float sigma = 1.0;

	float* data_in = new float[l];

	//Generate random data
	srand((unsigned)time(NULL)); 
	for (unsigned long z = 0; z < l; z++)
		data_in[z] = rand() % 9;

    	//Seed random distributions    	
	random_device rd;
	mt19937 seed(rd());

	//McKernel
	startTime = clock();
	McKernel* mckernel = FactoryMcKernel::createMcKernel(FactoryMcKernel::RBF, data_in, nv, dn, D, seed, sigma);
	mckernel->McFeatures();
	mckernel->McEvaluate();
	endTime = clock();

	cout << "RBF took " << double( endTime - startTime ) / double( CLOCKS_PER_SEC ) * 1000.0 << " ms." << endl;

	return 0;
}
