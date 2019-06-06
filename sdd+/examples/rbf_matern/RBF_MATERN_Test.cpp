/* McKernel: Approximate Kernel Expansions in Log-linear Time through Randomization		    

   Authors: Curtó and Zarza
   {curto,zarza}.2@my.cityu.edu.hk 						    */

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
	unsigned long t = 5;
	float sigma = 1.0;

	float* data_in = new float[l];

	//Generate random data
	srand((unsigned)time(NULL)); 
	for (unsigned long z = 0; z < l; z++)
		data_in[z] = rand() % 9;

    	//Seed random distributions    	
	random_device rd;
    	unsigned long seed = (unsigned long)rd();

	//McKernel
	startTime = clock();
	McKernel* mckernel = FactoryMcKernel::createMcKernel(FactoryMcKernel::MRBF, data_in, nv, dn, D, seed, sigma, t);
	mckernel->McFeatures();
	mckernel->McEvaluate();
	endTime = clock();
	cout << "RBF MATÉRN took " << double( endTime - startTime ) / double( CLOCKS_PER_SEC ) * 1000.0 << " ms." << endl;

	return 0;
}
