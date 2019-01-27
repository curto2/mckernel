/* McKernel: Approximate Kernel Expansions in Log-linear Time through Randomization		    

   Authors: Curtó and Zarza
   {curto,zarza}@estudiants.urv.cat 						    */

#ifndef FM_H
#define FM_H

#include "McKernel.hpp"

//McKernel  
class McKernel
{	
	public:
		unsigned long M_nv;
		unsigned long M_dn;
		unsigned long M_dnpg;
		unsigned long M_D;
		unsigned long M_dn_D;
		float* M_data;
		float* M_data_out;
		float* M_features;

       		vector<unsigned long> dl_Pi;
        	vector<float> dl_G;
        	vector<float> dl_C;
        	vector<long> dl_B; 

		McKernel( float* data, const unsigned long nv, const unsigned long dn, const unsigned long D, const unsigned long seed, const float sigma);

		virtual void McFeatures(); //Zx computation

		virtual void McEvaluate(); //Feature complex mapping

		virtual ~McKernel(){};
};

//RBF GAUSSIAN 
class RBF_GAUSSIAN : public McKernel
{
	public:
		RBF_GAUSSIAN(float* data, const unsigned long nv, const unsigned long dn, const unsigned long D, const unsigned long seed, const float sigma = 10.0);

		virtual ~RBF_GAUSSIAN(){};
};

//RBF MATÉRN
class RBF_MATERN : public McKernel
{
	public:
		RBF_MATERN(float* data, const unsigned long nv, const unsigned long dn, 
			const unsigned long D, const unsigned long seed, const float sigma = 1.0, const unsigned long t = 40);

		virtual ~RBF_MATERN(){};
};

//Factory McKernel
class FactoryMcKernel
{
	public: 
		enum TypeMcKernel 
		{
			RBF, //RBF GAUSSIAN
			MRBF  //RBF MATÉRN
		};

		static McKernel* createMcKernel(TypeMcKernel typemckernel, float* data, const unsigned long nv, 
			const unsigned long dn, const unsigned long D, const unsigned long seed, const float sigma = 1.0, const unsigned long t = 40);
};

#endif
