/* McKernel: A Library for Approximate Kernel Expansions in Log-linear Time 
   Curtó, Zarza, Yang, Smola, De La Torre, Ngo, and Van Gool 		    

   Authors: Curtó and Zarza
   {curto,zarza}@tinet.cat 						    */

#include <ctime>
#include <iostream>
#include "../../hpp/McKernel.hpp"

int main( int argc, char ** argv ) 
{
	// Initialize timing counter
	clock_t startTime, endTime;

	//Initialize variables
        long lt = 1UL << 21; 
        float* data = new float[lt];
	srand((unsigned)time(NULL)); 

	//Generate random data
	for (int i = 0; i < lt; i++)
		data[i] = rand() % 9 ;

	//FWHT
	startTime = clock();
	fwht(data, log2(lt));
	endTime = clock();
	cout << "FWHT lenght " << lt << " took " << double( endTime - startTime ) / double( CLOCKS_PER_SEC ) * 1000.0 << "ms" << endl;

	return 0;
}

