/* McKernel: Approximate Kernel Expansions in Log-linear Time through Randomization		    

   Authors: Curtó and Zarza
   {curto,zarza}.2@my.cityu.edu.hk 						    */

#include <ctime>
#include <iostream>
#include "../../hpp/McKernel.hpp"

int main( int argc, char ** argv ) 
{
	//Initialize timing counter
	clock_t startTime, endTime;

	//Initialize variables
	long lt = 1UL << 21; 
	float* data = new float[lt];
	srand((unsigned)time(NULL)); 

	//Generate random data
	for (int z = 0; z < lt; z++)
		data[z] = rand() % 9 ;

	//FWH
	startTime = clock();
	fwh(data, log2(lt));
	endTime = clock();
	cout << "FWH length " << lt << " took " << double( endTime - startTime ) / double( CLOCKS_PER_SEC ) * 1000.0 << " ms." << endl;

	return 0;
}

