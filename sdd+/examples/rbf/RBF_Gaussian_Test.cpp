/* McKernel: A Library for Approximate Kernel Expansions in Log-linear Time 
   Curtó, Zarza, Yang, Smola, De La Torre, Ngo, and Van Gool 		    

   Authors: Curtó and Zarza
   {curto,zarza}@tinet.cat 						    */

#include <ctime>
#include <iostream>
#include "../../hpp/Factory_McKernel.hpp"

int main()
{
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
	for (unsigned long i = 0; i < l; i++)
		data_in[i] = rand() % 9;

    	//Seed random distributions    	
	random_device rd;
    	unsigned long seed = (unsigned long)rd();

	//McKernel
	startTime = clock();
	McKernel* mckernel = FactoryMcKernel::createMcKernel(FactoryMcKernel::RBF, data_in, nv, dn, D, seed, sigma);
	mckernel->McFeatures();
	mckernel->McEvaluate();
	endTime = clock();

	cout << "RBF Gaussian took " << double( endTime - startTime ) / double( CLOCKS_PER_SEC ) * 1000.0 << "ms" << endl;

	return 0;
}
