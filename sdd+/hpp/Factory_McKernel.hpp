/* McKernel: A Library for Approximate Kernel Expansions in Log-linear Time 
   Curtó, Zarza, Yang, Smola, De La Torre, Ngo, and Van Gool 		    

   Authors: Curtó and Zarza
   {curto,zarza}@tinet.cat 						    */

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
        	vector<float> dl_S;
        	vector<long> dl_B; 

		McKernel( float* data, const unsigned long nv, const unsigned long dn, const unsigned long D, const unsigned long seed, const float sigma = 1.0);

		virtual void McFeatures(); //Vx computation

		virtual void McEvaluate(); //Feature complex mapping

		virtual ~McKernel(){};
};

//RBF Gaussian 
class RBF_Gaussian : public McKernel
{
	public:
		RBF_Gaussian(float* data, const unsigned long nv, const unsigned long dn, const unsigned long D, const unsigned long seed, const float sigma = 1.0);

		virtual ~RBF_Gaussian(){};
};

//RBF Matern
class RBF_Matern : public McKernel
{
	public:
		RBF_Matern(float* data, const unsigned long nv, const unsigned long dn, 
			const unsigned long D, const unsigned long seed, const float sigma = 1.0, const unsigned long t = 5);

		virtual ~RBF_Matern(){};
};

//Factory McKernel
class FactoryMcKernel
{
	public: 
		enum TypeMcKernel 
		{
			RBF, //RBF Gaussian
			MRBF  //RBF Matern
		};

		static McKernel* createMcKernel(TypeMcKernel typemckernel, float* data, const unsigned long nv, 
			const unsigned long dn, const unsigned long D, const unsigned long seed, const float sigma = 1.0, const unsigned long t = 5);
};

#endif
